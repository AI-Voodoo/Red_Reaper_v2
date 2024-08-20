import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt

from data_processing.data_loaders import GeneralFileOperations
from nlp.embeddings import EmbeddingModel

class Autoencoder(nn.Module):
    def __init__(self, input_dim=768):
        super(Autoencoder, self).__init__()
        
        # Define encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 32)  # Bottleneck layer
        )

        # Define decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)


class TrainAE:
    def __init__(self) -> None:
        self.file_ops = GeneralFileOperations()
        self.batch_size = 16
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Autoencoder(input_dim=768).to(self.device)
        self.embedding_work = EmbeddingModel()
        self.model_out_path = "data/model/model.pth"

    def AE_train_model(self):
        self.file_ops.delete_file(self.model_out_path)

        # Load datasets
        train_dataset, val_dataset = self.embedding_work.train_test_set_embeddings("data/stage_1/refined_high_value_emails.json")

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        criterion = nn.MSELoss()

        #, weight_decay=1e-5 & increase epoc max & lr increase from 0.01
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.8, verbose=True)
        
        min_val_loss = float('inf')
        early_stopping_counter = 0

        train_losses = []
        val_losses = []
        for epoch in range(10000): 

            self.model.train()
            epoch_train_loss = 0
            
            for batch_data in train_loader:
                batch_data = batch_data[0].to(self.device)
                
                optimizer.zero_grad() 
                outputs = self.model(batch_data)
                loss = criterion(outputs, batch_data)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())  # Record the batch loss
                
                epoch_train_loss += loss.item()
            
            epoch_train_loss /= len(train_loader)  # Average training loss for the epoch

            # Validation loss
            self.model.eval()
            epoch_val_loss = 0
            with torch.no_grad():
                for val_data in val_loader:
                    val_data = val_data[0].to(self.device)
                    val_outputs = self.model(val_data)
                    loss = criterion(val_outputs, val_data)
                    val_losses.append(loss.item())  # Record the batch loss
                    epoch_val_loss += loss.item()
            epoch_val_loss /= len(val_loader) 
            print(f"Epoch {epoch+1}, Train Loss: {epoch_train_loss:.7e}, Validation Loss: {epoch_val_loss:.7e}")

            # Update the scheduler with the current validation loss
            scheduler.step(epoch_val_loss)
            
            # Early stopping based on validation loss
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                torch.save(self.model.state_dict(), self.model_out_path)
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
            
            if early_stopping_counter == 5:
                print("Early stopping!")
                break

        # Plot the training and validation loss
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        return self.model
    

class InferenceAE:
    def __init__(self, model_path, input_dim=768):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Autoencoder(input_dim=input_dim).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.criterion = nn.MSELoss()
        self.embedding_work = EmbeddingModel()

    def run_inference(self):
        contents, embeddings_tensor = self.embedding_work.test_ae_classificaton_load_set()
        inference_data = []
        with torch.no_grad():
            for i, (content, embedding) in enumerate(zip(contents, embeddings_tensor)):
                embedding = embedding.to(self.device)
                reconstructed_embedding = self.model(embedding)
                loss = self.criterion(reconstructed_embedding, embedding).item()
                print(f"Content {i+1}: {content}")
                print(f"Loss: {loss:.7e}\n")
                inference_data.append({
                    "content": content,
                    "loss": loss  # Store the loss as a float for accurate sorting
                })
        
        # Sort the inference data by loss in ascending order
        sorted_inference_data = sorted(inference_data, key=lambda x: x['loss'])
        
        # Optionally, format the loss back to scientific notation for display purposes
        for data in sorted_inference_data:
            data['loss'] = f"{data['loss']:.7e}"
        
        return sorted_inference_data
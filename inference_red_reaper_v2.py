import numpy as np
from autoencoder.autoencoder import InferenceAE
from data_processing.data_loaders import GeneralFileOperations
from nlp.title import Repaer

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

class Classification:
    def __init__(self) -> None:
        pass
    
    def plot_confusion_matrix(self, true_labels, predicted_labels, classes, title):
        # Calculate the confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels, labels=classes)
        
        # Normalize the confusion matrix by row (i.e., by the number of samples in each class)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Use Seaborn to plot the normalized confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_normalized, annot=True, fmt=".2%", cmap='Blues', cbar=False, 
                    xticklabels=classes, yticklabels=classes, annot_kws={"size": 12})
        
        plt.title(title, fontsize=16)
        plt.ylabel('True label', fontsize=14)
        plt.xlabel('Predicted label', fontsize=14)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        # Colorbar setup can be added if needed for a specific range
        # plt.colorbar()
        plt.show()


file_ops = GeneralFileOperations()
run_ae = InferenceAE(model_path="data/model/model.pth")
classify = Classification()
title_repaer = Repaer()

enron_email_path = "data/enron_emails/emails.csv"
infernce_out_path = "data/stage_1/infernce_output.json"
infernce_out_high_path = "data/testing/high_infernce_output.json"
infernce_out_low_path = "data/testing/low_infernce_output.json"

gpt_eval_set_high = "data/gpt_test_set/hv.txt"
gpt_eval_set_low = "data/gpt_test_set/lv.txt"

training_samples = 37000
high_score_threshold=0.379
low_score_threshold=0.378
start_seed = 32
reconstruction_threshold_high = 0.00099999
reconstruction_threshold_low = 0.001
test_sample_amount = 100


title_repaer.print_title("Testing")

# select testing method
choice = input("\n\nChoose to test on:\n[1] Enron unseen data (random unseen sample, output to infernce_output.json)\n[2] Plot confusion matrix on GPT unseen data\n")

if choice == "1":
    # testing based on unseen enron data
    print("loading data...")
    run_ae.run_enron_random_sample_inference(enron_email_path, infernce_out_path, test_sample_amount, reconstruction_threshold_high, reconstruction_threshold_low, high_score_threshold, low_score_threshold, training_samples, start_seed)
elif choice == "2":
    # testing based on unseen gpt synth data
    print("loading data...")
    high_inference_data = run_ae.test_inference(gpt_eval_set_high, infernce_out_high_path, reconstruction_threshold_high, reconstruction_threshold_low, high_score_threshold, low_score_threshold)

    low_inference_data = run_ae.test_inference(gpt_eval_set_low, infernce_out_low_path, reconstruction_threshold_high, reconstruction_threshold_low, high_score_threshold, low_score_threshold)

    if high_inference_data and low_inference_data:
        # Combine high value and low value results into single lists
        all_true_labels = ['high value'] * len(high_inference_data) + ['low value'] * len(low_inference_data)
        all_predicted_labels_ae = [item['ae_class'] for item in high_inference_data + low_inference_data]
        all_predicted_labels_cosine = [item['cosine_class'] for item in high_inference_data + low_inference_data]

        # Class labels
        classes = ['high value', 'low value']

        # Plotting the confusion matrices
        classify.plot_confusion_matrix(all_true_labels, all_predicted_labels_ae, classes, 'Confusion Matrix for Autoencoder')
        classify.plot_confusion_matrix(all_true_labels, all_predicted_labels_cosine, classes, 'Confusion Matrix for Cosine Similarity')
else:
    print("no choice selected.")
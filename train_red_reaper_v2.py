from data_processing.data_loaders import GeneralFileOperations, LoadEmailData
file_ops = GeneralFileOperations()
file_ops.delete_file("data/logs/general_log.log")

from autoencoder.autoencoder import TrainAE
from data_processing.pre_process import DataPreProcess
load_data = LoadEmailData()
preprocess = DataPreProcess()
train_ae = TrainAE()

enron_email_path = "data/enron_emails/emails.csv"
training_set_path = "data/stage_1/training_set.json"

samples = 37000
score_threshold = 0.379

# stage 1: create inital dataset by finding potential training examples
#emails = load_data.load_process_email_data(samples, enron_email_path, score_threshold)


# stage 2: prepare training set
kept_emails, discarded_data = load_data.prepapre_training_set(score_threshold, data_path="data/stage_1/high_value_emails.json")
file_ops.save_data_to_json(kept_emails, training_set_path)
print(f"training set size: {len(kept_emails)}")


# stage 3: train AE
train_ae.AE_train_model(training_set_path)


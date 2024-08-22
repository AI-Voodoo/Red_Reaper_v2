from data_processing.data_loaders import GeneralFileOperations, LoadEmailData
from autoencoder.autoencoder import TrainAE
from data_processing.pre_process import DataPreProcess
from nlp.title import Repaer

file_ops = GeneralFileOperations()
load_data = LoadEmailData()
preprocess = DataPreProcess()
train_ae = TrainAE()
title_repaer = Repaer()

enron_email_path = "data/enron_emails/emails.csv"
data_create_training_set_path = "data/stage_1/high_value_emails.json"
training_set_path = "data/stage_1/training_set.json"

training_samples = 37000
min_score_threshold = 0.379
start_seed = 32

title_repaer.print_title("Training")

# stage 1: create inital dataset by finding potential training examples
print("loading data...")
emails = load_data.load_process_email_data(training_samples, enron_email_path, min_score_threshold, start_seed)


# stage 2: prepare training set
print("loading data...")
kept_emails, discarded_data = load_data.prepapre_training_set(min_score_threshold, data_create_training_set_path)
file_ops.save_data_to_json(kept_emails, training_set_path)
print(f"Built training set size: {len(kept_emails)}")


# stage 3: train AE
print("loading data...")
train_ae.AE_train_model(training_set_path)


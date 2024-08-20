from data_processing.data_loaders import GeneralFileOperations, LoadEmailData
file_ops = GeneralFileOperations()
file_ops.delete_file("data/logs/general_log.log")

from autoencoder.train_ae import TrainAE, InferenceAE
from data_processing.pre_process import DataPreProcess
load_data = LoadEmailData()
preprocess = DataPreProcess()
train_ae = TrainAE()
run_ae = InferenceAE(model_path="data/model/model.pth")

#import logging
#logging.basicConfig(filename='data/logs/general_log.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')


enron_email_path = "data/enron_emails/emails.csv"
refined_email_path = "data/stage_1/refined_high_value_emails.json"
infernce_path = "data/stage_1/infernce.json"

# stage 1: create inital dataset by finding potential training examples
emails = load_data.load_process_email_data(enron_email_path)


# stage 2: prepare training set
kept_emails, discarded_data = load_data.prepapre_training_set()
file_ops.save_data_to_json(kept_emails, refined_email_path)
print(f"training set size: {len(kept_emails)}")


# stage 3: train AE
train_ae.AE_train_model()


# stage 4: eval
inference_data = run_ae.run_inference()
file_ops.save_data_to_json(inference_data, infernce_path)


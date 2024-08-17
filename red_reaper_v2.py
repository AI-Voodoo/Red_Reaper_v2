from data_processing.data_loaders import GeneralFileOperations, LoadEmailData
file_ops = GeneralFileOperations()
file_ops.delete_file("data/logs/general_log.log")

from data_processing.pre_process import DataPreProcess
load_data = LoadEmailData()
preprocess = DataPreProcess()

import logging
logging.basicConfig(filename='data/logs/general_log.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')



#stage 1: create inital dataset by finding potential training examples
enron_email_path = "data/enron_emails/emails.csv"
refined_email_path = "data/stage_1/refined_high_value_emails.json"
#emails = load_data.load_process_email_data(enron_email_path)



# stage 2: refine dataset to pick the best subset of examples
kept_emails, discarded_data = load_data.filter_emails_based_on_alignment()
file_ops.save_data_to_json(kept_emails, refined_email_path)
print(f"training set size: {len(kept_emails)}")


# stage 3: LLM synthesize dataset based on best examples


# stage 4: train Autoencoder



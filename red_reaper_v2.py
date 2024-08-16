from data_processing.data_loaders import GeneralFileOperations, LoadEmailData
file_ops = GeneralFileOperations()
file_ops.delete_file("data/logs/general_log.log")

from data_processing.pre_process import DataPreProcess
load_data = LoadEmailData()
preprocess = DataPreProcess()

import logging
logging.basicConfig(filename='data/logs/general_log.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')



#stage 1:
enron_email_path = "data/enron_emails/emails.csv"
#emails = load_data.load_process_email_data(enron_email_path)



# stage 2:
kept_emails, discarded_data = load_data.filter_emails_based_on_alignment()

print("")

import logging
from data_processing.data_loaders import LoadData

logging.basicConfig(filename='data/logs/general_log.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')


load_data = LoadData()

enron_email_path = "data/enron_emails/emails.csv"
load_data.load_process_csv_data(enron_email_path)
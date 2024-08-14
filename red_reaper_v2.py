from data_processing.data_loaders import LoadData
load_data = LoadData()

load_data.delete_file("data/logs/general_log.log")
import logging


logging.basicConfig(filename='data/logs/general_log.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')




enron_email_path = "data/enron_emails/emails.csv"
load_data.load_process_csv_data(enron_email_path)
from data_processing.data_loaders import LoadData



load_data = LoadData()

enron_email_path = "data/enron_emails/emails.csv"
load_data.load_process_csv(enron_email_path)
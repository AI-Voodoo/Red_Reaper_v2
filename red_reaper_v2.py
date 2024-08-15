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
emails = load_data.load_process_email_data(enron_email_path)


# stage 2:
"""
emails = file_ops.load_json("data/stage_1/high_value_emails.json")
discarded_data = []
good_data = []
for email in emails:
    email_meta_data = email['email_meta_data']
    clean_content = email['clean_content']
    law_score = email['law_score']
    money_score = email['money_score']
    combined_score = email['combined_score']
    focused_law_content = email['focused_law_content']
    focused_money_content = email['focused_money_content']

    _, alpha_ratio, _ = preprocess.analyze_text(clean_content)
    if alpha_ratio < 0.75:
        discarded_data.append(email)
    else:
        good_data.append(email)


print("")
"""
import numpy as np
from autoencoder.autoencoder import InferenceAE
from data_processing.data_loaders import GeneralFileOperations


class Classification:
    def __init__(self) -> None:
        pass


file_ops = GeneralFileOperations()
run_ae = InferenceAE(model_path="data/model/model.pth")
classify = Classification()

enron_email_path = "data/enron_emails/emails.csv"
infernce_path = "data/stage_1/infernce_output.json"

samples = 37000
high_score_threshold=0.379
low_score_threshold=0.378

#inference_data = run_ae.run_enron_random_sample_inference(enron_email_path, infernce_path, sample_amount=100, high_value_threshold=0.00039, low_value_threshold=0.00075, cosine_threshold=score_threshold, seen_samples=samples)

high_inference_data = run_ae.test_inference("data/gpt_test_set/hv.txt", "data/testing/high_infernce_output.json", high_value_threshold=0.00099999, low_value_threshold=0.001, cosine_threshold=high_score_threshold, cosine_min_threshold=low_score_threshold)

low_inference_data = run_ae.test_inference("data/gpt_test_set/lv.txt", "data/testing/low_infernce_output.json", high_value_threshold=0.00099999, low_value_threshold=0.001, cosine_threshold=high_score_threshold, cosine_min_threshold=low_score_threshold)


print()

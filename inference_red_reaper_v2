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

samples = 20000
score_threshold=0.379

inference_data = run_ae.run_inference(enron_email_path, infernce_path, sample_amount=100, high_value_threshold=0.00039, low_value_threshold=0.00075, cosine_threshold=score_threshold, seen_samples=samples)

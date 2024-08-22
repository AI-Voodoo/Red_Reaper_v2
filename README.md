# Red Reaper v2
<p align="center">
<img src="https://github.com/AI-Voodoo/Red_Reaper_v2/blob/main/data/images/rr.v2.png?raw=true" alt="Description" style="width:100%;" />
</p>

## Global Project Overview
The Red Reaper project is all about automation identifying sensitive communications that could be weaponized for malicious purposes. In [ Red Reaper v1](https://www.cybermongol.ca/frontier-research/red-reaper-building-an-ai-espionage-agent), the system demonstrated its ability to operationalize stolen messages, using an LLM to devise strategies and analyze and map communication patterns using graph data science. However, the true potential of this system depends on a critical first stage: accurately classifying valuable emails.

Red Reaper v2 takes this essential capability to the next level by incorporating deep learning. This version focuses on training the system to distinguish valuable communications—those that can be exploited—from the rest. The early results, as shown in the confusion matrices, are promising.

Here's the evolution:
- v1 was static, relying on an ensemble of inference methods, using NER classification and targeted embedding similarities to identify high-value emails. While effective, it was limited.

- v2 introduces a learning component, initially trained on the Enron email dataset and tested against GPT-generated synthetic communications (small set for now). Now, the system can not only generalize by identifying unseen, valuable emails—but can also improve over time.

**This project is still in its early proof-of-concept phase**, but I’ll be sharing write-ups and code as development progresses. The rest of this repo focusses on stage 1, malicious email/chat classification. The ultimate goal of this project is to spark curiosity and inspire others to build and explore the intersection of cyber security and data science. 


<b>*Reminder: Ensure to install cuda version of pytorch if you can - things will be much faster.*</b>
- This was developed using dual A 6000’s (96GB vRAM) / 256GB CPU RAM
- This was also tested on a decent gaming laptop (8GB vRAM) / 64GB CPU RAM




## Stage 1: Sensitive Email/Chat Classification for Adversary Agent Emulation (espionage)


### Training 
<p align="left">
<img src="https://github.com/AI-Voodoo/Red_Reaper_v2/blob/main/data/images/training.PNG?raw=true" alt="Description" style="width:100%;" />
</p>

*Note: Training is optional as is this repo comes with a model for inference. But, just doing inference is no fun…let’s dive in and see how this was trained.*

### Building the Training Set
[Download Enron email corpus](https://www.kaggle.com/datasets/wcukierski/enron-email-dataset) and save to [data/enron_emails/](https://github.com/AI-Voodoo/Red_Reaper_v2/tree/main/data/enron_emails).

Before we can train a model to classify potentially valuable communications from an espionage agent’s perspective, we need to build a training dataset. This means either labeling examples ourselves of what we consider high-value communications, or—more efficiently—using automation to do the heavy lifting.

That’s right, we’re letting automation handle the initial labeling. We’re using an autoencoder, a type of deep neural network designed for anomaly detection, which will be trained on a single class. Our plan is to sift through the half-million emails in the Enron corpus, which range from highly confidential enterprise communications to personal emails about an employee's love life and create a training subset. This diverse mix means our automated method for identifying sensitive corporate communications has a serious challenge ahead.

At the heart of this process, we [define two key strings that semantically represent the legal and financial](https://github.com/AI-Voodoo/Red_Reaper_v2/blob/main/nlp/prompts.py) undertones we hope the agent will identify as high-value. We then use sentence transformers to generate embeddings—numerical representations—of these strings and compare them with embeddings for every email in the corpus using cosine similarity. Emails that surpass a certain threshold make it into the training set for our autoencoder.

This method can be extended by defining additional target strings such as those related to IT systems, credentials etc. 

### Training an Autoencoder
First, what is an autoencoder? For this, I will let my friends at MIT deep dive on it – I strongly recommend this video to get a feel for what is happening here.

**watch:** [MIT 6.S191 (2023): Deep Generative Modeling](https://www.youtube.com/watch?v=3G5hWM6jqPk&t=662s)

Now that we have some background, we're going to feed the embeddings created from the training set into the autoencoder. We'll use a [network architecture](https://github.com/AI-Voodoo/Red_Reaper_v2/blob/main/autoencoder/autoencoder.py) that encodes these embeddings, learning to keep only the most salient features in a latent space, while discarding the rest. When the autoencoder decodes these features from the latent space, it evaluates how well it can reconstruct the embedding compared to the original, using a loss function. This process trains the autoencoder to identify the most important semantic features of the text that define sensitive communications.

### Evaluating the Autoencoder
<p align="left">
<img src="https://github.com/AI-Voodoo/Red_Reaper_v2/blob/main/data/images/testing.PNG?raw=true" alt="Description" style="width:100%;" />
</p>

Now that we have a trained model—or if you're using the pre-trained model included with the repo—we can evaluate its performance. Classification is achieved by passing data through the trained autoencoder and then observing the error value. Lower error values indicate that the autoencoder recognizes the data, while higher error values suggest it does not. A threshold is set to determine the classification. This is a straightforward setup to demonstrate the concept.

We have two evaluation options for this:
1.	**Random Sampling of Unseen Enron Data:** This method allows us to compare how a cosine similarity threshold classifies the text versus how the autoencoder does. This quick evaluation provides an [initial look](https://github.com/AI-Voodoo/Red_Reaper_v2/blob/main/data/stage_1/infernce_output.json) at the model's performance, offering insights into its effectiveness. json sorted in DESC order, HIGH -> LOW.
2.	**Plotting Confusion Matrices:** This approach involves testing the model against a small set of GPT-generated high-value and low-value classes. Although the test set is small, it seems to do the trick. I also chose to include shorter-length strings in the test set, as this might challenge the model more, given the nature of the training data. Additionally, I wanted to assess whether the model could be used to classify shorter messages, such as those found in Teams or Slack messages.

### Sample High/Low-Value Classification on the Enron Dataset
The high-value email reveals an impending FBI raid on the facilities—information that criminals could exploit significantly before it becomes public. In contrast, the low-value email is simply a conversation about hang gliding.

**HIGH: FBI Raid**
<p align="left">
<img src="https://github.com/AI-Voodoo/Red_Reaper_v2/blob/main/data/images/FBI_high_value.PNG?raw=true" alt="Description" style="width:100%;" />
</p>

**LOW: Hang Gliding**
<p align="left">
<img src="https://github.com/AI-Voodoo/Red_Reaper_v2/blob/main/data/images/hang_gliding_low_value.PNG?raw=true" alt="Description" style="width:100%;" />
</p>

**Confusion Matrix**
<p align="left">
<img src="https://github.com/AI-Voodoo/Red_Reaper_v2/blob/main/data/plots/ae_c_matrix.png?raw=true" alt="Description" style="width:100%;" />
</p>


### Coming
In the next stage (Stage 2), we will setup a local LLM to take these classified sensitive communications and create operationalizing strategies.


# Red Reaper v2
<p align="center">
<img src="https://github.com/AI-Voodoo/Red_Reaper_v2/blob/main/data/images/rr.v2.png?raw=true" alt="Description" style="width:100%;" />
</p>

## Global Project Overview
The Red Reaper project is all about automation identifying email communications that could be weaponized for malicious purposes. In [ Red Reaper v1](https://www.cybermongol.ca/frontier-research/red-reaper-building-an-ai-espionage-agent), the system demonstrated its ability to operationalize stolen messages, using an LLM to devise strategies and analyze and map communication patterns using graph data science. However, the true potential of this system depends on a critical first stage: accurately classifying valuable emails.

Red Reaper v2 takes this essential capability to the next level by incorporating deep learning. This version focuses on training the system to distinguish valuable communications—those that can be exploited—from the rest. The early results, as shown in the confusion matrices, are promising.

Here's the evolution:
- v1 was static, relying on an ensemble of inference methods, using NER classification and targeted embedding similarities to identify high-value emails. While effective, it was limited.

- v2 introduces a learning component, initially trained on the Enron email dataset and tested against GPT-generated synthetic communications (small set for now). Now, the system can not only generalize by identifying unseen, valuable emails—but can also improve over time.

**This project is still in its early proof-of-concept phase**, but I’ll be sharing write-ups and code as development progresses. The rest of this repo focusses on stage 1, malicious email/chat classification. The ultimate goal of this project is to spark curiosity and inspire others to build and explore the intersection of cyber security and data science. 


<b>*Reminder: Ensure to install cuda version of pytorch if you can - things will be much faster.*</b>
- This was developed using dual A 6000’s (96GB vRAM) / 256GB CPU RAM
- This was also tested on a decent gaming laptop (8GB vRAM) / 64GB CPU RAM




## Stage 1: Malicious Email/Chat Classification for Red Team Operations


### Training 
<p align="left">
<img src="https://github.com/AI-Voodoo/Red_Reaper_v2/blob/main/data/images/training.PNG?raw=true" alt="Description" style="width:100%;" />
</p>

*Note: Training is optional as is this repo comes with a model for inference. But, just doing inference is no fun…let’s dive in and see how this was trained.*

### Building the Training Set
[Download Enron email corpus](https://www.kaggle.com/datasets/wcukierski/enron-email-dataset) and save to [data/enron_emails/](https://github.com/AI-Voodoo/Red_Reaper_v2/tree/main/data/enron_emails)
Before we can train a model to classify potentially valuable communications from a red team perspective, we need to construct a training dataset. 
 

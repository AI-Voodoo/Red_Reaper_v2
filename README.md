# Red Reaper v2

## Global Project Overview
The Red Reaper project is all about automation identifying email communications that could be weaponized for malicious purposes. In [ Red Reaper v1](https://www.cybermongol.ca/frontier-research/red-reaper-building-an-ai-espionage-agent), the system demonstrated its ability to operationalize stolen messages, using an LLM to devise strategies and analyze and map communication patterns using graph data science. However, the true potential of this system depends on a critical first step: accurately triaging valuable emails.

Red Reaper v2 takes this essential capability to the next level by incorporating deep learning. This version focuses on training the system to distinguish valuable communications—those that can be exploited—from the rest. The early results, as shown in the confusion matrices, are promising.

Here's the evolution:
- v1 was static, relying on an ensemble of inference methods, using NER classification and targeted embedding similarities to identify high-value emails. While effective, it was limited.

- v2 introduces a learning component, initially trained on the Enron email dataset and tested against GPT-generated synthetic communications (small set for now). Now, the system can not only generalize by identifying unseen, valuable emails—but can also improve over time.

This project is still in its proof-of-concept phase, but I’ll be sharing write-ups and code as development progresses. The goal is to spark curiosity and inspire others to build and explore. Look for the first release in the next few days. 


## Stage 1: Malicious Email/Chat Classification for Red Team Operations

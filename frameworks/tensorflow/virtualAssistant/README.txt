# Paper: Automatic Extraction of Opinion-based Q&A from Online Developer Chats
============================================================================================
In the following, we briefly describe the different components that are included in this project and the softwares required to run the experiments.


## Project Structure
The project includes the following files and folders:

  - __/data__: A folder that contains inputs and outputs that are used for the experiments
	- question: development set, test set for opinion-asking question identification. Also contains files used for evaluating comparison techniques DECA and SentiStrengthSE.
	- answer: development set, training set and test set for answer extraction.
	- WordEmbeddings: 200 dimensional GloVe word vectors trained on Stack Overflow and public chats(IRC and Slack).
			
  - __/scripts__: Contains scripts for running the experiments
	- Opinion-asking question Identification using patterns (identify_rec_patterns.py) 
	- Answer Extraction using Deep Learning. Contains supporting input file and helper scripts, entry point of the experiment (main.py)
	- Evaluation of comparison techniques of Opinion-asking question Identification (ques_baselines.py), Answer Extraction (ans_heuristics.py). ans_heuristics.py also generates an .arff file which could be used as input to Weka for replicating the ML-based answer extraction.


## Software Requirements
Provided in the requirements.txt


## Running Experiments
Step1: Install software requirements mentioned above.
Step2: Run scripts:
a) Opinion-asking Question Identification: run identify_rec_patterns.py
b) Answer Extraction and model traing: main.py 





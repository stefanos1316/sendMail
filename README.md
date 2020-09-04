# machine_learning_frameworks_analysis
Here we store everything related to the "An analysis on the energy and run-time performance of machine learning frameworks".
Below there are some points that we could fill in in order to help us in this research.


# Motivation
According to [1] the energy needed to train AI is now rising seven times
faster than ever before.
Therefore, selecting the proper algorithms to train and predict
could significantly reduce the monetary and environmental
impact of the IT sector.
Such a choice is even more paramount since the data-sets
are always increasing and more training is required to
precisely predict.


# Related Work
Works done in the past three years (2018 and 2019) from [top-tier ML conferences](http://www.guide2research.com/topconf/machine-learning):

[NeurIPS](https://papers.nips.cc/):
* [E2-Train: Training State-of-the-art CNNs with Over 80% Energy Savings](https://papers.nips.cc/paper/8757-e2-train-training-state-of-the-art-cnns-with-over-80-energy-savings)

[CVPR](https://openaccess.thecvf.com/menu):
None related study was found for this conference.

[ICML](http://proceedings.mlr.press/)
None related study was found for this conference.

[ACL](https://www.aclweb.org/portal/acl)
* [Energy and Policy Considerations for Deep Learning in NLP](https://arxiv.org/abs/1906.02243)

[AAAI](https://aaai.org/Library/conferences-library.php)
None related study was found for this conference.

[ICSE](https://2019.icse-conferences.org/program/program-icse-2019)
* [What can Android mobile app developers do about the energy consumption of machine learning?](https://softwareprocess.es/pubs/mcintosh2018EMSE-MLEnergy.pdf)

[FSE]
None related study was found for this conference


[2] discusses methods on how to obtain energy measurements correctly
from a machine learning algorithm.
[3] investigates the energy requirements of CNN and compares
its energy/accurasy. Moreover, it provides a detailed workload characterization
to facilitate the design of energy efficient deep learning solutions.
[4]


# Research Questions

Primary RQs to be answered:
* Which are the most energy-efficient machine learning
frameworks?---Here we investigate the overall
energy performance of the selected ML framework by examining
the following steps:
	* Performance implications of the Training
	* Performance implications of the Prediction

* Which are the most run-time performance-efficient machine learning
frameworks?---Similarly to the above, we investigate the overall
run-time performance of the selected ML framework by examining
the following steps:
	* Performance implications of the Training
	* Performance implications of the Prediction

* Which machine learning algorithms more energy
and run-time performance-efficient for the selected tasks?---Here
we try to find out which ML algorithms, among the selected
ML frameworks, perform better for particular tasks.

Secondary RQs in case we have time:
* How much can the hyperparameter tuning affect the energy
and run-time performance of the correspondning
Machine Learning Frameworks?---Here we are going
to examine a number of tunable hyperparameters to find their
energy and run-time performance implications.


# Copmuter Platform
[Stereo](https://www.dell.com/downloads/global/products/pedge/t320_spec_sheet.pdf) equipped with [Nvidia RTX 2080 Ti](https://lambdalabs.com/blog/2080-ti-deep-learning-benchmarks/). 


# Frameworks

Here we can set a number of selection criteria
such as the following:

* Open Source
* Activity (last commit no more than a year)
* Popularity (based on [GitHub stars](https://github.com/aymericdamien/TopDeepLearning)
* Evaluation (evaluated in related work)
* Documentation
* Programming Language (in order to be fair to the ML frameworks, we select only the Python framework as Python is the most popular PL for ML)
* Working toy tutorial
* Evaluation on the same data (data used in one framework are applicable to all the other frameworks)

Top GitHub frameworks for Machine Learning based on stars:
* Tensorflow
* Keras
* opencv
* PyTorch
* Bert


# Supervised Algorithms
There are two categories of machine learning algorithms
we can test: (1) standard machine learning algorithms (e.g., linear regression, naive bayes, k-means)
and (2) deep learning algoriths (e.g., Feedforward Neural Networks, Convolutional Neural Networks, Recurrent Neural Networks).
I am not sure how much time it will take to implement just one of this
using a framework.
However, we need to find a method on how to select the best candidates.


# Tasks
An idea is to search among the MSR data-set papers and select that candidate tasks
for deep learning based on the available and well-documented ones.


# References
[1] @article{SGM_2019,
	title = {Energy and {Policy} {Considerations} for {Deep} {Learning} in {NLP}},
	url = {http://arxiv.org/abs/1906.02243},
	urldate = {2020-08-29},
	journal = {arXiv:1906.02243 [cs]},
	author = {Strubell, Emma and Ganesh, Ananya and McCallum, Andrew},
	month = jun,
	year = {2019},
	note = {arXiv: 1906.02243},
	keywords = {Computer Science - Computation and Language},
	annote = {Comment: In the 57th Annual Meeting of the Association for Computational Linguistics (ACL). Florence, Italy. July 2019},
}

[2] @article{GRRG_2019,
	title = {Estimation of energy consumption in machine learning},
	volume = {134},
	issn = {0743-7315},
	url = {http://www.sciencedirect.com/science/article/pii/S0743731518308773},
	doi = {10.1016/j.jpdc.2019.07.007},
	language = {en},
	urldate = {2020-08-29},
	journal = {Journal of Parallel and Distributed Computing},
	author = {García-Martín, Eva and Rodrigues, Crefeda Faviola and Riley, Graham and Grahn, Håkan},
	month = dec,
	year = {2019},
	keywords = {Deep learning, Energy consumption, GreenAI, High performance computing, Machine learning},
	pages = {75--88},
}

[3] @inproceedings{LCBZ_2016,
	title = {Evaluating the {Energy} {Efficiency} of {Deep} {Convolutional} {Neural} {Networks} on {CPUs} and {GPUs}},
	doi = {10.1109/BDCloud-SocialCom-SustainCom.2016.76},
	booktitle = {2016 {IEEE} {International} {Conferences} on {Big} {Data} and {Cloud} {Computing} ({BDCloud}), {Social} {Computing} and {Networking} ({SocialCom}), {Sustainable} {Computing} and {Communications} ({SustainCom}) ({BDCloud}-{SocialCom}-{SustainCom})},
	author = {Li, Da and Chen, Xinbo and Becchi, Michela and Zong, Ziliang},
	month = oct,
	year = {2016},
	keywords = {Biological neural networks, central processing unit, CPU, deep convolutional neural networks, deep learning, deep learning framework, energy conservation, Energy consumption, energy efficiency, Energy efficiency, energy-efficiency, GPU, GPUs, graphics processing unit, graphics processing units, Graphics processing units, Hardware, learning (artificial intelligence), Machine learning, neural nets, neural networks, Training},
	pages = {477--484},
}


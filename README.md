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
[2] discusses methods on how to obtain energy measurements correctly
from a machine learning algorithms.
[3] investigates the energy requirements of CNN and compares
the energy/accurasy. Moreover, it provides a detailed workload characterization
to facilitate the design of energy efficient deep learning solutions.
[4]



# Research Questions
* What are the energy and run-time performance implications
of different Machine Learning algorithms?---Here we are trying
to investigate the overall performance of different algorithms
to find out which are the most energy- and run-time performance-efficient ones.
* What are the energy and run-time performance implications
of the selected Machine Learning frameworks?---Here we are
going to examine two different sub-research questions
in order to identify the performance implications
of each different steps.
	* Performance implications of the Training
	* Performance implications of the Prediction
* How much can the hyperparameter tuning affect the energy and run-time performance
of the correspondning Machine Learning Frameworks?---Here we are going
to examine a number of tunable hyperparameters to find their
energy and run-time performance implications.
* Are specific machine learning algorithms more energy
and run-time performance efficient for specific tasks?---Here
we are examining whether the are algorithms, among the selected
ML frameworks, that perform better for particular tasks than others.


# Copmuter Platform
[Stereo](https://www.dell.com/downloads/global/products/pedge/t320_spec_sheet.pdf) equipped with [Nvidia RTX 2080 Ti](https://lambdalabs.com/blog/2080-ti-deep-learning-benchmarks/). 

# Frameworks
Here we have to choose 4-5 since it is going to take too much time
to train the models.
I think we can use [GitHub stars](https://github.com/aymericdamien/TopDeepLearning)
to select the TOP 4-5 frameworks.
By executing the python2.7 scripts/generate_stats.py you can get a list with the top starred
deep learning projects on GitHub
However, if you have another opinion please share.

Top GitHub frameworks for Machine Learning:
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
This is a challenging step!
Why should find out and implemented the same tasks for
each of the framworks that we are going to use.
An example of task is the Spam filter detection for emails
were we feed our model/nueral network with a number of
data to train it and then we can pass inputs to predict
the result.
We should also consider having small, medium,
and large tasks.
For instance, recognize an image can be a small task.


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


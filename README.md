# machine_learning_frameworks_analysis

Here we store everything related to the "An analysis on the energy and run-time performance of machine learning frameworks".
Below there are some points that we could fill in in order to help us in this research.


# Installation

First, install and create the virtual environment.
Note that the linux-tools-`uname`-generic cannot be install on OSX or Windows-like machines.
This tools is deticated to linux and facilitates the extraction of energy measurements from RAPL interface.

```bash
apt-get install python3-venv linux-tools-`uname -r`-generic
git lfs install
pip3 install pre-commit freeze
pre-commit install
pre-commit migrate-config
cd frameworks
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

To exit from the virtual environment just execute the `deactive` command.
To install all the necesary python packages execute the `pip3 install -e ".[deploy]"`
After installing all the python packages you can start executing
the deep learning algorithms.


# Optional steps to get energy measurements
Before executing the experiments make sure ot run the commands below or add them inside a script.
These commands are needed to execute sudo-less `perf` to fetch energy measurements from RAPL interface.


```bash
sudo sh -c 'echo -1 >/proc/sys/kernel/perf_event_paranoid'
sudo sysctl -w kernel.perf_event_paranoid=-1
sudo bash framework/governor.sh pe
```


# Contribution instructions
* To install packages and test models, always use the virtual environment (`source env/bin/activate`)
* Make sure you added the required packages inside the `framework/requirements.txt` to faciliate the reproducability of our work (execute `pip freeze > requirements.txt` inside the frameworks direcotry)
* Add many comments in your code
* Files larger than 100 MiB should be stores in the LFS system


# Motivation

According to [1] the energy needed to train AI is now rising seven times
faster than ever before.
Therefore, selecting the proper algorithms to train and predict
could significantly reduce the monetary and environmental
impact of the IT sector.
Such a choice is even more paramount since the data-sets
are always increasing and more training is required to
precisely predict.
Moreover, once a model is trained, it is then used multiple
of times in the inference phase.
Therefore, it is important to evaluate the performance of the inference phase too.

# Related Work

## Conferences

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
* [ReluDiff: Differential Verification of Deep Neural Networks](https://arxiv.org/pdf/2001.03662.pdf)
* [Improving the Effectiveness of Traceability Link Recoveryusing Hierarchical Bayesian Networks](https://arxiv.org/pdf/2005.09046.pdf)
* [Understanding the Automated Parameter Optimization onTransfer Learning for Cross-Project Defect Prediction: An Empirical Study](http://taochen.github.io/publications/papers/camera-ready.pdf)
* [Big Code != Big Vocabulary:Open-Vocabulary Models for Source Code](http://homepages.inf.ed.ac.uk/s1467463/documents/icse20-main-1325.pdf)
* [An Empirical Study on Program Failures of Deep Learning Jobs](http://hongyujohn.github.io/icse20-main-199.pdf)
* [SEWordSim: software-specific word similarity database](https://dl.acm.org/doi/abs/10.1145/2591062.2591071)

[FSE](https://2021.esec-fse.org/)
* [A Comprehensive Study on Challenges in Deploying Deep Learning Based Software](https://2020.esec-fse.org/details/fse-2020-papers/92/A-Comprehensive-Study-on-Challenges-in-Deploying-Deep-Learning-Based-Software)
* [A comprehensive study on deep learning bug characteristics](https://dl.acm.org/doi/pdf/10.1145/3338906.3338955)

[ICCSA 2016]()
* [A Public Bug Database of GitHub Projects and its Application in Bug Prediction](https://www.researchgate.net/publication/304664263_A_Public_Bug_Database_of_GitHub_Projects_and_Its_Application_in_Bug_Prediction)


[2] discusses methods on how to obtain energy measurements correctly
from a machine learning algorithm.
[3] investigates the energy requirements of CNN and compares
its energy/accurasy. Moreover, it provides a detailed workload characterization
to facilitate the design of energy efficient deep learning solutions.
[10] has a data-set of Java bugs collected from the GitHub.
The authors of the corresponding paper has used this [data-set](https://www.inf.u-szeged.hu/~ferenc/papers/GitHubBugDataSet/)
to train 13 machine learning algorithms.
Another data-set used widely by researcher to detect duplicate defects reports is this [one](https://github.com/logpai/bugrepo).

## Journals

[EMSE](https://www.springer.com/journal/10664)
* [What Makes a Popular Academic AI Repository?](https://arxiv.org/abs/2010.02472)

[ACM Computing Surveys](https://dl.acm.org/journal/csur)
* [A Survey of Machine Learning for Big Code and Naturalness](https://dl.acm.org/doi/abs/10.1145/3212695)

IJSEA
* [Benchmarking Machine Learning Techniques for software defect detection](https://arxiv.org/pdf/1506.07563.pdf)

[TSE](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=32)
* [SEthesaurus: WordNet in Software Engineering](https://ieeexplore.ieee.org/abstract/document/8827962)

## Misc

* [Predicting Software Effort Estimation Using Machine Learning Techniques](https://ieeexplore.ieee.org/abstract/document/8486222)


# Research Questions

Primary RQs to be answered:
1. Which are the most energy-efficient machine learning
frameworks?---Here we investigate the overall
energy performance of the selected ML framework by examining
the following steps:
	* Performance implications of the Training
	* Performance implications of the Prediction

TS: Do you want to pick one task out of the selected tasks (3?) for this RQ? Or, all the tasks can be considered here (in that case, we dont need the third RQ - task specific performance).
SG: About the 3rd I was thinking that it can be interesting to find out which framework perform better for a task and find out why. But on a second thought, we can do also such an analysis
in the RQ 1. and 2., no?

2. Which are the most run-time performance-efficient machine learning
frameworks?---Similarly to the above, we investigate the overall
run-time performance of the selected ML framework by examining
the following steps:
	* Performance implications of the Training
	* Performance implications of the Prediction

3. What are the energy, run-time performance, and accuracy
trade-offs of the examined ML frameworks?---Here we are going
to investigate what are each of the frameworks performance
trade-offs with respect to the energy, run-time performance,
and accuracy. Here we could also present a ranking
so that developers could select among frameworks or algorithms
to develop a task whenever energy, run-time performance,
or accuracy is of high demand.

* Which machine learning algorithms are more energy
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

The actual hardware is not the Stereo but a new machine that is far stronger.
Panos informed me that they are still expecting the motherboard to come.


# Frameworks

Here we can set a number of selection criteria
such as the following:

* Open Source (All four)
* Activity (last commit no more than a year) (All four)
* Popularity (based on GitHub stars)
* Evaluation (evaluated in related work [6]--[9]) 
* Documentation (All four)
* Working toy tutorial (All four)
* Evaluation on the same data (data used in one framework are applicable to all the other frameworks)
* General tasks (select frameworks that can implement different tasks and not specific) (All four)

To collect our candidates, we first used a script found in the [TopDeepLearning repository](https://github.com/aymericdamien/TopDeepLearning).
The corresponding repository offers a Python script that performs a search on GitHub repository Titles and Project descriptions
fields to obtain and rank the repositories with the most GitHub stars based on a number of keywords.
We modified the corresponding Python script's keyworks because some popular ML frameworks
were missing from the results.
After executing the script, we ended with the following list of [repositories](methods/top_ml_frameworks_initial.md) (executing 2020-10-18)
From the initial list, we first excluded repositories that were related to learning courses, papers, etc.
and we ended up with the following [list](methods/top_ml_frameworks_remove_unrelated.md).

Top GitHub frameworks for Machine Learning based on stars:
* Tensorflow
* PyTorch
* Scikit-learn (Although we may use it for deep learning, may forums do not suggest it as it is not supporting GPU usage)
* Caffe

TS: One concern here is that not all framework/libraries support all kinds of tasks. for example, opencv is a vision library; hence tasks other than image/vision might not be feasible.
Two alternatives to mitigate this: 1) choose only generic frameworks 2) task specific frameworks.

SG: I think both of your above points are interesting to examine. However, I think not many people will choose to implement specific task
with specific framework in order to gain more energy savings, but instead they will choose a single framework that is overall more efficient.
So I think we should stick with the (1) and choose more generic frameworks.


# Tasks
There is a really nice and diverse [data-set](https://github.com/tensorflow/datasets) for research purposes
available under the Tensorflow repository [5].
Since it has many tasks for each of the selected categories, we may consider
selecting 1 from each, but which? :P

TS: Potential candidates:

	* NLP-based - [IMDB reviews sentiment analysis](https://www.tensorflow.org/datasets/catalog/imdb_reviews) or [Text summarization](https://www.tensorflow.org/datasets/catalog/multi_news)
	* [image classification](https://www.tensorflow.org/datasets/catalog/cifar10),
 	* [regression](https://www.tensorflow.org/datasets/catalog/forest_fires)
        * defect prediction data-set ([PROMISE](http://promise.site.uottawa.ca/SERepository/datasets-page.html), [NASA](https://github.com/klainfo/NASADefectDataset))
 
MK: We need to find tasks that are already implemented (and are open source software).
Then, we need to write these tasks into other frameworks.
Let's focus on Python-based frameworks as a starting point.
IMHO, it would be interesting to look at NLP tasks.

TS: Agree with MK

MK: Other benchmarks that could be useful for NNs are CIFAR-10 (also mentioned by TS), CIFAR-100, and MNIST.

SG: I have already found some implementations of the above DNN architectures. I am running some tests on Stereo, but I believe we can find most of the on GitHub.


# Algorithms
There are two categories of machine learning algorithms
we can test: (1) standard machine learning algorithms (e.g., linear regression, naive bayes, k-means)
and (2) deep learning algoriths (e.g., Feedforward Neural Networks, Convolutional Neural Networks, Recurrent Neural Networks).
First, we will focus on deep learning algorithms and if we have more time (or maybe extend it for a journal)
then we can use standard machine learning algorithms.
To select the popular deep learning algorithms for each task,
we first collected all the machine learning paper published in ICSE starting from 2020.
Then, we classify for each task the used deep learning algorithms.
The online sheet `algorithm_from_icse` to see the most popular deep learning algorithms
for each task we defined in the previous section.


# Goal

* Find the popular deep learning algorithms for each task
* Implemented tasks use the above algorithms with Tensorflow and Pytorch
* Created script to execute tasks and collect measurements
* Analyze results
* Start with paper writing


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

[5] @misc{TFDS,
  title = {{TensorFlow Datasets}, A collection of ready-to-use datasets},
  howpublished = {\url{https://www.tensorflow.org/datasets}},
}

[6] @incollection{paszke_pytorch_2019,
	title = {{PyTorch}: {An} {Imperative} {Style}, {High}-{Performance} {Deep} {Learning} {Library}},
	shorttitle = {{PyTorch}},
	url = {http://papers.nips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf},
	urldate = {2020-10-18},
	booktitle = {Advances in {Neural} {Information} {Processing} {Systems} 32},
	publisher = {Curran Associates, Inc.},
	author = {Paszke, Adam and Gross, Sam and Massa, Francisco and Lerer, Adam and Bradbury, James and Chanan, Gregory and Killeen, Trevor and Lin, Zeming and Gimelshein, Natalia and Antiga, Luca and Desmaison, Alban and Kopf, Andreas and Yang, Edward and DeVito, Zachary and Raison, Martin and Tejani, Alykhan and Chilamkurthy, Sasank and Steiner, Benoit and Fang, Lu and Bai, Junjie and Chintala, Soumith},
	editor = {Wallach, H. and Larochelle, H. and Beygelzimer, A. and Alché-Buc, F. d{\textbackslash}textquotesingle and Fox, E. and Garnett, R.},
	year = {2019},
	pages = {8026--8037},
	file = {NIPS Snapshot:/home/sgeorgiou/Zotero/storage/5HHDN2XL/9015-pytorch-an-imperative-stylehigh-.html:text/html}
}

[7] @inproceedings{abadi_tensorflow_2016,
	title = {{TensorFlow}: {A} {System} for {Large}-{Scale} {Machine} {Learning}},
	isbn = {978-1-931971-33-1},
	shorttitle = {{TensorFlow}},
	url = {https://www.usenix.org/conference/osdi16/technical-sessions/presentation/abadi},
	language = {en},
	urldate = {2020-10-18},
	author = {Abadi, Martin and Barham, Paul and Chen, Jianmin and Chen, Zhifeng and Davis, Andy and Dean, Jeffrey and Devin, Matthieu and Ghemawat, Sanjay and Irving, Geoffrey and Isard, Michael and Kudlur, Manjunath and Levenberg, Josh and Monga, Rajat and Moore, Sherry and Murray, Derek G. and Steiner, Benoit and Tucker, Paul and Vasudevan, Vijay and Warden, Pete and Wicke, Martin and Yu, Yuan and Zheng, Xiaoqiang},
	year = {2016},
	pages = {265--283},
	file = {Full Text PDF:/home/sgeorgiou/Zotero/storage/Y6C7E5UG/Abadi et al. - 2016 - TensorFlow A System for Large-Scale Machine Learn.pdf:application/pdf;Snapshot:/home/sgeorgiou/Zotero/storage/ENW6ZDMR/abadi.html:text/html}
}

[8] @article{pedregosa_scikit-learn_2011,
	title = {Scikit-learn: {Machine} {Learning} in {Python}},
	volume = {12},
	issn = {1533-7928},
	shorttitle = {Scikit-learn},
	url = {http://jmlr.org/papers/v12/pedregosa11a.html},
	number = {85},
	urldate = {2020-10-18},
	journal = {Journal of Machine Learning Research},
	author = {Pedregosa, Fabian and Varoquaux, Gaël and Gramfort, Alexandre and Michel, Vincent and Thirion, Bertrand and Grisel, Olivier and Blondel, Mathieu and Prettenhofer, Peter and Weiss, Ron and Dubourg, Vincent and Vanderplas, Jake and Passos, Alexandre and Cournapeau, David and Brucher, Matthieu and Perrot, Matthieu and Duchesnay, Édouard},
	year = {2011},
	pages = {2825--2830},
	file = {Full Text PDF:/home/sgeorgiou/Zotero/storage/XW5Q59L8/Pedregosa et al. - 2011 - Scikit-learn Machine Learning in Python.pdf:application/pdf;Snapshot:/home/sgeorgiou/Zotero/storage/JCX8CN6X/pedregosa11a.html:text/html}
}

[9] @inproceedings{jia_caffe_2014,
	address = {New York, NY, USA},
	series = {{MM} '14},
	title = {Caffe: {Convolutional} {Architecture} for {Fast} {Feature} {Embedding}},
	isbn = {978-1-4503-3063-3},
	shorttitle = {Caffe},
	url = {https://doi.org/10.1145/2647868.2654889},
	doi = {10.1145/2647868.2654889},
	urldate = {2020-10-18},
	booktitle = {Proceedings of the 22nd {ACM} international conference on {Multimedia}},
	publisher = {Association for Computing Machinery},
	author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
	month = nov,
	year = {2014},
	keywords = {computer vision, machine learning, neural networks, open source, parallel computation},
	pages = {675--678}
}

[10] @inproceedings{toth_public_2016,
	address = {Cham},
	series = {Lecture {Notes} in {Computer} {Science}},
	title = {A {Public} {Bug} {Database} of {GitHub} {Projects} and {Its} {Application} in {Bug} {Prediction}},
	isbn = {978-3-319-42089-9},
	doi = {10.1007/978-3-319-42089-9_44},
	language = {en},
	booktitle = {Computational {Science} and {Its} {Applications} -- {ICCSA} 2016},
	publisher = {Springer International Publishing},
	author = {Tóth, Zoltán and Gyimesi, Péter and Ferenc, Rudolf},
	editor = {Gervasi, Osvaldo and Murgante, Beniamino and Misra, Sanjay and Rocha, Ana Maria A.C. and Torre, Carmelo M. and Taniar, David and Apduhan, Bernady O. and Stankova, Elena and Wang, Shangguang},
	year = {2016},
	keywords = {Bug database, Bug prediction},
	pages = {625--638},
	file = {Submitted Version:/home/sgeorgiou/Zotero/storage/YX75HRQQ/Tóth et al. - 2016 - A Public Bug Database of GitHub Projects and Its A.pdf:application/pdf}
}


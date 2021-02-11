import pandas as pd, re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pycorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('http://localhost:9000')
from sklearn.metrics import precision_recall_fscore_support

data = pd.read_csv('../../data/question/test_set.csv')
grouped_conv = data.groupby(['thread'], sort=False)


def deca():
    print ("DECA")
    deca = pd.read_csv('../../data/question/test_DECA.csv')
    true = deca['is_rec_question'].values.tolist()
    pred = deca['DECA (1- Opinion Asking, 0- Otherwise)'].values.tolist()
    eval(true, pred)


def sentiStrengthSE():
    print("SentiStrengthSE")
    # Generated using SentiStrengthSE trinary sentiment classification
    sentiSE = pd.read_csv('../../data/question/test_SentiStrengthSE.csv')
    true = sentiSE['is_rec_question'].values.tolist()
    pred = sentiSE['sentiment'].values.tolist()
    pred = [1 if x==-1 else x for x in pred] #considering both negative and positive polarities
    eval(true, pred)


def NLTK(text):
    sid = SentimentIntensityAnalyzer()
    ss = sid.polarity_scores(text)
    if (ss['neg'] > 0.0) or (ss['pos'] > 0.0):
        sent_score = 1
    else:
        sent_score = 0
    return sent_score


def CoreNLP(text):
    # start server
    # cd stanford-corenlp-full-2016-10-31/
    # java -mx5g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -timeout 10000
    result = nlp.annotate(text, properties={'annotators': 'sentiment', 'outputFormat': 'json', 'timeout': '1000000'})
    for s in result['sentences']:
        score = (s['sentimentValue'], s['sentiment'])
    # List of sentiments: Verynegative | Negative | Neutral | Positive | Verypositive
    if score[1] != 'Neutral':
        sent_score = 1
    else:
        sent_score = 0
    return sent_score


def sentiTools(mode):
    print (mode)
    df = pd.read_csv('../../data/question/test_GroundTruth.csv')
    true = df['is_rec_question'].values.tolist()
    pred = []
    messages = df['message'].values.tolist()

    for msg in messages:
        # preprocess to remove urls and code
        msg = str(msg)
        msg = re.sub('<http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'url', msg)  # Replace urls
        msg = re.sub('```([^`]*)```', 'code', msg)  # Replace multiline code
        msg = re.sub('`([^`]*)`', 'code', msg)  # Replace any code
        if mode == "NLTK":
            sentiment = NLTK(msg)
        else:
            sentiment = CoreNLP(msg)
        pred.append(sentiment)
    eval(true, pred)


def eval(true, pred):
    precision, recall, fscore, support = precision_recall_fscore_support(true, pred, average='binary')
    print('precision: {:.2f}'.format(precision))
    print('recall: {:.2f}'.format(recall))
    print('fscore: {:.2f}'.format(fscore))
    print ("="*50)


if __name__ == '__main__':
    deca()
    sentiStrengthSE()
    sentiTools('NLTK')
    sentiTools('CoreNLP')

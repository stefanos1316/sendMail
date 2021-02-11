import pandas as pd, numpy as np, re
from seqeval.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
from pycorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('http://localhost:9000')


def pred2binarylabel(pred):
    idx2tag = {0: 'NonAnswer', 1: 'Answer'}
    out = []
    for pred_i in pred:
        if pred_i >= 1:
            out.append(idx2tag[1])
        else:
            out.append(idx2tag[0])
    return out


def baseline_location(grouped_by_conv):
    print("HE:location")
    test_Y = []
    pred_Y = []

    for thread, conv in grouped_by_conv:
        preds = []
        labels = []
        count = 0
        for row, data in conv.iterrows():
            is_rec = data[3]

            if is_rec == 1:
                labels.append(0)
                preds.append(0)
            else:
                labels.append(is_rec)
                if count == 1:
                    preds.append(1)
                else:
                    preds.append(0)
            count += 1

        test_Y.append(labels)
        pred_Y.append(preds)

    # flatten the lists
    test_Y = [item for sublist in test_Y for item in sublist]
    pred_Y = [item for sublist in pred_Y for item in sublist]
    test_Y = [1 if x == 2 else x for x in test_Y]
    eval(test_Y, pred_Y)
    return test_Y, pred_Y


def load_word_embeddings():
    # Load our customized GloVe embeddings trained on SO
    embeddings_index = {}
    f = open('../../data/WordEmbeddings/SO_vectors.txt', 'rb')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index


def calc_avg_embedding(text, embeddings_index):
    # creating average word vectors for each utterance
    count = 0
    words = text.split(" ")
    vec = np.zeros(200).reshape((1, 200))
    for word in words:
        try:
            vec += embeddings_index.get(bytes(word, 'utf-8')).reshape((1, 200))
            count += 1
        except:
            continue
    if count != 0:
        vec /= count
    return vec


def calc_sim_score(ques, ans, embeddings_index):
    # Calculating textual similarity of question and answer candidate
    ques_vector = calc_avg_embedding(ques, embeddings_index)
    ans_vector = calc_avg_embedding(ans, embeddings_index)
    if cosine_similarity(ques_vector, ans_vector)[0][0] > 0.5:
        sim = 1
    else:
        sim = 0
    return sim


def calc_sentiment(text):
    # custom preprocess
    text = str(text)
    text = re.sub('<http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'url',
                     text)  # Replace urls
    text = re.sub('<@[A-Z][a-z]*>', ' username', text)  # Replace user mentions
    text = re.sub(':[^:\s]*(?:::[^:\s]*)*:', 'emoji', text)  # Replace emoji
    text = re.sub('’', '\'', text)  # handle utf encodings
    text = re.sub('```([^`]*)```', 'code', text)  # Replace only multiline code
    text = re.sub('`([^`]*)`', 'code', text)  # Replace any code

    # Determining sentiment of answer candidate
    # Using Stanford CoreNLP
    # start server
    # cd stanford-corenlp-full-2016-10-31/
    # java -mx5g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -timeout 10000
    result = nlp.annotate(text, properties={'annotators': 'sentiment','outputFormat': 'json','timeout': '10000000'})
    for s in result['sentences']:
        score = (s['sentimentValue'], s['sentiment'])
    # List of sentiments: Verynegative | Negative | Neutral | Positive | Verypositive
    if score[1] != 'Neutral':
        sent_score = 1
    else:
        sent_score = 0
    return sent_score


def baseline_text(grouped_by_conv, mode):
    print ("HE:"+str(mode))
    embeddings_index = load_word_embeddings()
    test_Y = []
    pred_Y = []

    for thread, conv in grouped_by_conv:
        scores = []
        labels = []
        for row, data in conv.iterrows():
            message = str(data[2])
            is_rec = data[3]

            if is_rec == 1:
                labels.append(0)
                scores.append(0)
                question = message
            else:
                candidate_ans = message
                if mode == "content":
                    score = calc_sim_score(question, candidate_ans, embeddings_index)
                else:
                    score = calc_sentiment(candidate_ans)
                scores.append(score)
                if is_rec == 2:
                    labels.append(1)
                else:
                    labels.append(0)
        test_Y.append(labels)
        pred_Y.append(scores)

    # flatten the lists
    test_Y = [item for sublist in test_Y for item in sublist]
    pred_Y = [item for sublist in pred_Y for item in sublist]
    eval(test_Y, pred_Y)
    return test_Y, pred_Y


# Writing data to arff file compatible with Weka
def write_arff(features):
    with open('test_ML.arff', "w") as fp:
        fp.write('''@RELATION Chats 

@ATTRIBUTE Location {1, 0}
@ATTRIBUTE Content {1, 0}
@ATTRIBUTE Sentiment {1, 0}
@ATTRIBUTE Answer {0, 1}
@DATA
''')
        for conv in features:
            fp.write(str(conv).replace('(','').replace(')','') + "\n")
    fp.close()


def eval(true, pred):
    pred_labels = pred2binarylabel(pred)
    true_labels = pred2binarylabel(true)
    print(classification_report(true_labels, pred_labels))
    print ("="*50)


if __name__ == '__main__':
    df = pd.read_csv('../../data/answer/test_set.csv')
    grouped_by_conv = df.groupby(['thread'], sort=False)

    print("Running Individual Heuristics")
    test_Y, loc_Y = baseline_location(grouped_by_conv)
    test_Y, cont_Y = baseline_text(grouped_by_conv, "content")
    test_Y, sent_Y = baseline_text(grouped_by_conv, "sentiment")

    print("Combining Heuristics and writing to arff files")
    features = zip(loc_Y, cont_Y, sent_Y, test_Y)
    write_arff(features)
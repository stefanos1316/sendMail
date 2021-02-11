import re, string
import numpy as np, pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from expand_contractions import *


def text_preprocess(message):
    message = str(message)
    message = re.sub('<http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'url',
                     message)  # Replace urls
    message = re.sub('<@[A-Z][a-z]*>', ' username', message)  # Replace user mentions
    message = re.sub(':[^:\s]*(?:::[^:\s]*)*:', 'emoji', message)  # Replace emoji
    message = re.sub('’', '\'', message)  # handle utf encodings
    message = expand_contractions(message)  # Expand contractions
    message = re.sub('```([^`]*)```', 'code', message)  # Replace only multiline code
    message = re.sub('`([^`]*)`', 'code', message) # Replace any code
    message = message.lower()  # Lower the text
    message = message.translate(str.maketrans('', '', string.punctuation)) # remove punctuation
    return message


def get_GloVe_embeddings():
    # Load our customized GloVe embeddings trained on SO and put in Data Folder
    embeddings_index = {}
    f = open('../../../data/WordEmbeddings/SO_vectors.txt','rb')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index


def get_embedding_matrix(embeddings_index, word_index, num_words, embedding_dim, max_features):
    # first create a matrix of zeros, this is our embedding matrix
    embedding_matrix = np.zeros((num_words, embedding_dim))

    # for each word in out tokenizer lets try to find that work in our w2v model
    for word, i in word_index.items():
        if i > max_features:
            continue
        embedding_vector = embeddings_index.get(bytes(word, 'utf-8'))
        if embedding_vector is not None:
            # add word vector to the matrix
            embedding_matrix[i] = embedding_vector
        else:
            # if word doesn't exist, assign a random vector within our embedding range
            embedding_matrix[i] = np.random.uniform(low=0.001, high=0.009, size=(embedding_dim,))
    return embedding_matrix


def get_conv_len(df):
    conv_lengths = []
    grouped_by_conv = df.groupby(['thread'], sort=False)
    for thread, conv in grouped_by_conv:
        conv_lengths.append(conv.shape[0])
    return conv_lengths


def create_utterance_embeddings(X, embedding_matrix):
    X_repr = []
    for utterance in X:
        vec = []
        for word in utterance:
            try:
                vec.append(embedding_matrix[word])
            except KeyError:
                continue
        X_repr.append(vec)
    X_repr = np.asarray(X_repr)
    return X_repr


def pad_nested_sequences(sequences, padding_len):
    pad_seq = []
    for seq in sequences:
        x = []
        count = 0
        vec = np.zeros(200)
        for word in seq:
            if count < padding_len:
                x.append(word)
                count += 1
        if len(x) < padding_len:
            for i in range(padding_len - len(x)):
                x.append(vec)
        pad_seq.append(x)
    pad_seq = np.asarray(pad_seq)
    return pad_seq


def create_sets(df, tokenizer, embedding_matrix):
    # Data pre-processing
    df['text'] = df['message'].apply(text_preprocess)
    # this takes our sentences and replaces each word with an integer
    X = tokenizer.texts_to_sequences(df['text'].values)
    # Create embeddings
    X0 = create_utterance_embeddings(X, embedding_matrix)
    # pad utterances to the same length
    X1 = pad_nested_sequences(X0, 20)
    Y1 = df['is_rec'].values
    X2 = []
    Y2 = []
    count = 0
    # get all conversation lengths
    conv_lengths = get_conv_len(df)
    sample_weights = []
    # Creating joint QA representations
    for conv in conv_lengths:
        x_temp = []
        y_temp = []
        weights = []
        for length in range(conv):
            if length == 0:
                ques_repr = X1[count]
            else:
                candidate_ans_repr = X1[count]
                concat_QA = np.concatenate((ques_repr, candidate_ans_repr), axis=0)
                x_temp.append(concat_QA)

                if Y1[count] == 2:
                    y_temp.append(1)
                    weights.append(1.0)
                else:
                    y_temp.append(0)
                    weights.append(1.0)
            count += 1
        X2.append(x_temp)
        Y2.append(y_temp)
        sample_weights.append(weights)

    X2 = np.asarray(X2)
    # Padding data for LSTM
    X2 = pad_sequences(X2, maxlen=20, padding="post", truncating="post", dtype='float32', value=0.)
    Y2 = pad_sequences(Y2, maxlen=20, padding="post", truncating="post")
    sample_weights = pad_sequences(sample_weights, maxlen=20, padding="post", truncating="post", value=0.0)
    Y2 = np.asarray(Y2)
    Y2 = Y2.reshape(Y2.shape[0], Y2.shape[1], 1)
    return X2, Y2, sample_weights


def prep_data(df_train, df_test, embedding_dim):
    df = pd.concat([df_train, df_test])
    # Data pre-processing
    df['text'] = df['message'].apply(text_preprocess)
    # Build vocabulary
    max_features = 30000  # number of words we care about
    tokenizer = Tokenizer(num_words=max_features, split=' ', oov_token='<unw>')
    tokenizer.fit_on_texts(df['text'].values)
    word_index = tokenizer.word_index
    print('Found %s unique tokens in vocab' % len(word_index))
    num_words = min(max_features, len(word_index)) + 1
    # get all GloVe embeddings
    embeddings_index = get_GloVe_embeddings()
    # get embedding matrix for words in our vocab
    embedding_matrix = get_embedding_matrix(embeddings_index, word_index, num_words, embedding_dim, max_features)
    X_train, Y_train, sample_weights_train = create_sets(df_train, tokenizer, embedding_matrix)
    X_test, Y_test, sample_weights_test = create_sets(df_test, tokenizer, embedding_matrix)
    return X_train, Y_train, X_test, Y_test, sample_weights_train, sample_weights_test



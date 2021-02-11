import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

# from keras.callbacks import EarlyStopping
import time

# from matplotlib import pyplot
from numpy.random import seed
import tensorflow as tf

seed(1)
tf.random.set_seed(2)
start_time = time.time()


def write_csv(test_labels, pred_labels):
    for list in test_labels:
        list.insert(0, "Question")
    for list in pred_labels:
        list.insert(0, "Question")
    test_file = "../../../data/answer/test_set.csv"
    test_df = pd.read_csv(test_file)
    grouped_by_conv = test_df.groupby(["thread"], sort=False)
    count = 0

    for thread, conv in grouped_by_conv:
        if conv.shape[0] != len(test_labels[count]):
            for i in range(conv.shape[0] - len(test_labels[count])):
                test_labels[count].append(0)
                pred_labels[count].append(0)
        count += 1

    test_list = [item for sublist in test_labels for item in sublist]
    pred_list = [item for sublist in pred_labels for item in sublist]
    test_df["TrueLabels"] = test_list
    test_df["PredLabels"] = pred_list
    test_df.to_csv("test_ChatE_pred.csv", index=False)


def calc_eval_measures():
    print("Evaluation Measures:")
    df = pd.read_csv("./test_ChatE_pred.csv")
    true = df["TrueLabels"].values.tolist()
    true = [x for x in true if x != "Question"]  # discarding question instances
    true = ["1" if x == "Answer" else x for x in true]  # converting to binary values
    true = ["0" if x == "NonAnswer" else x for x in true]  # converting to binary values

    pred = df["PredLabels"].values.tolist()
    pred = [x for x in pred if x != "Question"]  # discarding question instances
    pred = ["1" if x == "Answer" else x for x in pred]  # converting to binary values
    pred = ["0" if x == "NonAnswer" else x for x in pred]  # converting to binary values

    precision, recall, fscore, support = precision_recall_fscore_support(
        true, pred, average="binary", pos_label="1"
    )
    print("precision: {:.2f}".format(precision))
    print("recall: {:.2f}".format(recall))
    print("fscore: {:.2f}".format(fscore))


def pred2binarylabel(pred, sample_weights):
    idx2tag = {0: "NonAnswer", 1: "Answer"}
    out = []
    for i, pred_i in enumerate(pred):
        out_i = []
        for j, p in enumerate(pred_i):
            if p >= 0.5 and (sample_weights[i][j] == 1.0):
                out_i.append(idx2tag[1])
            elif p < 0.5 and (sample_weights[i][j] == 1.0):
                out_i.append(idx2tag[0])
        out.append(out_i)
    return out


def train_test_validate(
    model, X_tr, Y_tr, X_te, Y_te, sample_weights_train, sample_weights_test
):
    # Fit the model on train data
    # early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    # history = model.fit(X_tr, Y_tr, batch_size=256, epochs=100, validation_split=0.2, callbacks=[early_stopping])
    model.fit(X_tr, Y_tr, batch_size=256, epochs=100, validation_split=0.2)
    #  pyplot.plot(history.history["loss"], label="train_loss")
    # pyplot.plot(history.history["val_loss"], label="val_loss")
    # pyplot.legend()
    # pyplot.show()

    # Evaluate the model on test data
    scores = model.evaluate(X_te, Y_te, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))
    test_pred = model.predict(X_te, verbose=1)
    pred_labels = pred2binarylabel(test_pred, sample_weights_test)
    test_labels = pred2binarylabel(Y_te, sample_weights_test)
    print("Model took", time.time() - start_time, "to run")
    write_csv(test_labels, pred_labels)  # write predicted outputs
    calc_eval_measures()  # calculate evaluation measures

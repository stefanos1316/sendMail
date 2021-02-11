import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Reshape,
    TimeDistributed,
    Conv2D,
    MaxPool2D,
    Concatenate,
    Flatten,
    Dropout,
    Bidirectional,
    LSTM,
    Dense,
)
from tensorflow.keras.models import Model
from data_helper import prep_data
from evaluation import train_test_validate
from numpy.random import seed

# from keras.optimizers import adam

seed(1)
tf.random.set_seed(2)

# Very few comments, just adding this to activete the pre-commit hook black and flake8
if __name__ == "__main__":
    df_train = pd.read_csv("../../../data/answer/train_set.csv")
    df_test = pd.read_csv("../../../data/answer/test_set.csv")

    embedding_dim = 200  # dim of our learned embedding model
    (
        X_train,
        Y_train,
        X_test,
        Y_test,
        sample_weights_train,
        sample_weights_test,
    ) = prep_data(df_train, df_test, embedding_dim)
    utterance_length = 40  # avg QA-combination utterance length
    conversation_length = 20  # avg conversation length
    inputs = Input(
        shape=(
            (conversation_length),
            utterance_length,
            embedding_dim,
        )
    )

    # CNN Layer
    num_filters = 50
    reshape = Reshape((conversation_length, utterance_length, embedding_dim, 1))(inputs)
    conv_0_3 = TimeDistributed(
        Conv2D(num_filters, kernel_size=(2, embedding_dim), activation="relu"),
        input_shape=(1, conversation_length, utterance_length, embedding_dim, 1),
    )(reshape)
    maxpool_0_3 = TimeDistributed(MaxPool2D(pool_size=(2, 1), padding="valid"))(
        conv_0_3
    )
    conv_1_3 = TimeDistributed(
        Conv2D(num_filters, kernel_size=(3, embedding_dim), activation="relu"),
        input_shape=(1, conversation_length, utterance_length, embedding_dim, 1),
    )(reshape)
    maxpool_1_3 = TimeDistributed(MaxPool2D(pool_size=(2, 1), padding="valid"))(
        conv_1_3
    )
    conv_2_3 = TimeDistributed(
        Conv2D(num_filters, kernel_size=(4, embedding_dim), activation="relu"),
        input_shape=(1, conversation_length, utterance_length, embedding_dim, 1),
    )(reshape)
    maxpool_2_3 = TimeDistributed(MaxPool2D(pool_size=(3, 1), padding="valid"))(
        conv_2_3
    )
    concatenated_tensor = Concatenate(axis=2)([maxpool_0_3, maxpool_1_3, maxpool_2_3])
    flatten = TimeDistributed(Flatten())(concatenated_tensor)
    output = Dropout(0.5)(flatten)

    # biLSTM Layer
    bilstm = Bidirectional(
        LSTM(units=200, return_sequences=True, recurrent_dropout=0.1)
    )(
        output
    )  # variational biLSTM)
    outputs = TimeDistributed(Dense(1, activation="sigmoid"))(bilstm)
    model = Model(inputs=inputs, outputs=outputs)

    opt = tf.keras.optimizers.Adam(lr=0.001)
    # model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["binary_accuracy"], sample_weight_mode='temporal')
    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=["binary_accuracy"],
        sample_weight_mode="temporal",
    )
    print(model.summary())

    # Evaluation
    train_test_validate(
        model,
        X_train,
        Y_train,
        X_test,
        Y_test,
        sample_weights_train,
        sample_weights_test,
    )

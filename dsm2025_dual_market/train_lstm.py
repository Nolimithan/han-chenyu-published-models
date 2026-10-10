"""Rolling predictor used for the DSM experiments.

The paper text names this stage BiGRU. The notebooks that carry the paper's
experimental settings (binary labels, 231-wide VMD input, sequence length 3,
65 percent training window, roll of 120 days, Adam learning rate as configured
per asset, stop when cross-entropy falls below 0.05) construct a unidirectional
LSTM cell with LeakyReLU. This file keeps that cell.

Requires TensorFlow 1.15.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

from . import config


def leaky_relu(x, leak=0.5, name="LeakyRelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * tf.abs(x)


def make_sequences(features, labels, seq_length):
    data_x, data_y = [], []
    for i in range(0, len(labels) - seq_length):
        data_x.append(features[i : i + seq_length])
        data_y.append(labels[i + seq_length])
    return data_x, data_y


def rolling_predict(
    features,
    labels,
    train_ratio=config.ASSETS["SSEC"]["train_ratio"],
    learning_rate=config.ASSETS["SSEC"]["learning_rate"],
    seed=config.SEED,
):
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    if features.shape[1] != config.DATA_DIM:
        raise ValueError(
            "expected %d VMD features, got %d" % (config.DATA_DIM, features.shape[1])
        )
    if labels.shape[1] != config.OUTPUT_DIM:
        raise ValueError("labels must be one-hot with 2 columns")

    tf.set_random_seed(seed)
    tf.reset_default_graph()
    data_x, data_y = make_sequences(features, labels, config.SEQ_LENGTH)
    train_size = int(len(data_y) * train_ratio)
    iteration = int((len(data_y) - train_size) / config.PERIOD)

    x_ph = tf.placeholder(tf.float32, [None, config.SEQ_LENGTH, config.DATA_DIM])
    y_ph = tf.placeholder(tf.float32, [None, config.OUTPUT_DIM])
    cell = tf.nn.rnn_cell.LSTMCell(num_units=config.HIDDEN_DIM, activation=leaky_relu)
    outputs, _states = tf.nn.dynamic_rnn(cell, x_ph, dtype=tf.float32)
    y_pred = tf.contrib.layers.fully_connected(outputs[:, -1], config.OUTPUT_DIM, activation_fn=None)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_ph, logits=y_pred))
    train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    current = 0
    test_predict = []
    test_true = []
    for step in range(iteration):
        train_x = np.array(data_x[current : train_size + current])
        train_y = np.array(data_y[current : train_size + current])
        if step + 1 == iteration:
            test_x = np.array(data_x[train_size + current : len(data_y)])
            test_y = np.array(data_y[train_size + current : len(data_y)])
        else:
            test_x = np.array(data_x[train_size + current : train_size + current + config.PERIOD])
            test_y = np.array(data_y[train_size + current : train_size + current + config.PERIOD])
        test_true.append(test_y)
        current += config.PERIOD
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            while True:
                _, step_loss = sess.run([train, loss], feed_dict={x_ph: train_x, y_ph: train_y})
                if config.STOP_LOSS > step_loss:
                    break
            test_predict.append(sess.run(y_pred, feed_dict={x_ph: test_x}))

    predicted = np.argmax(np.vstack(test_predict), axis=1)
    actual = np.argmax(np.vstack(test_true), axis=1)
    report = classification_report(actual, predicted, target_names=["action", "inaction"])
    scores = {
        "precision_macro": precision_score(actual, predicted, average="macro"),
        "recall_macro": recall_score(actual, predicted, average="macro"),
        "f1_macro": f1_score(actual, predicted, average="macro"),
        "report": report,
    }
    return predicted, actual, scores


def save_predictions(path, actual, predicted):
    pd.DataFrame({"observation": actual, "prediction": predicted}).to_csv(path, index=False)

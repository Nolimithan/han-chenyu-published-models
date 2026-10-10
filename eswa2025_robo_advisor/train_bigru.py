"""BiGRU predictor on the 200-dimensional DCNN features.

Sequence length 3, hidden size 200, three-class softmax, Adam at 0.005.
Each roll retrains a fixed-length window and tests the next 90 days, stopping
when cross-entropy falls below 0.05. The placeholder keep_prob in the archived
notebook is not connected to the GRU graph, so no dropout is applied here.
The paper text states a dropout rate of 0.3.

Requires TensorFlow 1.15.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from .config import BIGRU


def make_sequences(features, labels, seq_length):
    data_x, data_y = [], []
    for i in range(0, len(labels) - seq_length):
        data_x.append(features[i : i + seq_length])
        data_y.append(labels[i + seq_length])
    return data_x, data_y


def rolling_predict(features, labels, train_ratio, seed=BIGRU["seed"]):
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    tf.set_random_seed(seed)
    tf.reset_default_graph()
    data_x, data_y = make_sequences(features, labels, BIGRU["seq_length"])
    train_size = int(len(data_y) * train_ratio)
    iteration = int((len(data_y) - train_size) / BIGRU["period"])

    x_ph = tf.placeholder(tf.float32, [None, BIGRU["seq_length"], features.shape[1]])
    y_ph = tf.placeholder(tf.float32, [None, BIGRU["output_dim"]])
    outputs, _ = tf.nn.bidirectional_dynamic_rnn(
        tf.nn.rnn_cell.GRUCell(num_units=BIGRU["hidden_dim"]),
        tf.nn.rnn_cell.GRUCell(num_units=BIGRU["hidden_dim"]),
        x_ph,
        dtype=tf.float32,
    )
    merged = tf.concat(outputs, 2)
    logits = tf.contrib.layers.fully_connected(merged[:, -1], BIGRU["output_dim"], activation_fn=None)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_ph, logits=logits))
    train = tf.train.AdamOptimizer(BIGRU["learning_rate"]).minimize(loss)

    current = 0
    test_predict, test_true = [], []
    for step in range(iteration):
        train_x = np.array(data_x[current : train_size + current])
        train_y = np.array(data_y[current : train_size + current])
        start = train_size + current
        if step + 1 == iteration:
            test_x = np.array(data_x[start:])
            test_y = np.array(data_y[start:])
        else:
            test_x = np.array(data_x[start : start + BIGRU["period"]])
            test_y = np.array(data_y[start : start + BIGRU["period"]])
        test_true.append(test_y)
        current += BIGRU["period"]
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            while True:
                _, step_loss = sess.run([train, loss], feed_dict={x_ph: train_x, y_ph: train_y})
                if BIGRU["stop_loss"] > step_loss:
                    break
            test_predict.append(sess.run(logits, feed_dict={x_ph: test_x}))

    predicted = np.argmax(np.vstack(test_predict), axis=1)
    actual = np.argmax(np.vstack(test_true), axis=1)
    binary_true = label_binarize(actual, classes=[0, 1, 2])
    binary_pred = label_binarize(predicted, classes=[0, 1, 2])
    scores = {
        "precision_macro": precision_score(actual, predicted, average="macro", zero_division=0),
        "recall_macro": recall_score(actual, predicted, average="macro", zero_division=0),
        "f1_macro": f1_score(actual, predicted, average="macro", zero_division=0),
        "precision_weighted": precision_score(actual, predicted, average="weighted", zero_division=0),
        "recall_weighted": recall_score(actual, predicted, average="weighted", zero_division=0),
        "f1_weighted": f1_score(actual, predicted, average="weighted", zero_division=0),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mae": float(mean_absolute_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
        "report": classification_report(actual, predicted, target_names=["sell", "hold", "buy"], zero_division=0),
    }
    try:
        scores["roc_auc_macro"] = float(roc_auc_score(binary_true, binary_pred, average="macro"))
    except ValueError:
        scores["roc_auc_macro"] = float("nan")
    return predicted, actual, scores


def save_predictions(path, actual, predicted, close=None):
    frame = pd.DataFrame({"observation": actual, "prediction": predicted})
    if close is not None:
        frame["close"] = np.asarray(close)
    frame.to_csv(path, index=False)

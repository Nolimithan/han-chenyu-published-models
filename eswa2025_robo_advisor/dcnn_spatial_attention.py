"""DCNN with the spatial attention module in Equation (7).

The executed graph is: Conv 32, 3x3, stride 1, LeakyReLU, spatial attention;
Conv 64, 3x3, stride 1, LeakyReLU, spatial attention; max-pool 2x2, stride 2;
flatten; fully connected layer of 200 units. Those 200 units are the features
passed to BiGRU. A softmax head is used only to train the extractor.

Channel attention and the full CBAM block are defined in an earlier notebook
cell and are not called by this graph. They are omitted.

The archived extractor is fit on the whole labeled sample, and the features
of the step before the loss falls below 0.01 are exported. BiGRU then makes
the train/test split. This file keeps that order.

Requires TensorFlow 1.15.
"""

import numpy as np
import pandas as pd
import tensorflow as tf

from .config import DCNN, IMAGE_COLS, IMAGE_ROWS


def leaky_relu(x, leak=DCNN["leak"], name="LeakyRelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * tf.abs(x)


def spatial_attention(input_feature):
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    avg_pool = tf.reduce_mean(input_feature, axis=[3], keep_dims=True)
    max_pool = tf.reduce_max(input_feature, axis=[3], keep_dims=True)
    concat = tf.concat([avg_pool, max_pool], 3)
    attention = tf.layers.conv2d(
        concat,
        1,
        (7, 7),
        padding="same",
        activation=tf.nn.sigmoid,
        kernel_initializer=kernel_initializer,
    )
    return attention * input_feature


def build_graph():
    x_dim = IMAGE_ROWS * IMAGE_COLS
    x_ph = tf.placeholder(tf.float32, [None, x_dim])
    y_ph = tf.placeholder(tf.float32, [None, 3])
    keep_prob = tf.placeholder(tf.float32)
    image = tf.reshape(x_ph, [-1, IMAGE_ROWS, IMAGE_COLS, 1])
    conv1 = tf.layers.conv2d(
        image, DCNN["conv1_filters"], [DCNN["kernel"], DCNN["kernel"]],
        strides=1, padding="same", activation=leaky_relu,
    )
    conv1 = spatial_attention(conv1)
    conv2 = tf.layers.conv2d(
        conv1, DCNN["conv2_filters"], [DCNN["kernel"], DCNN["kernel"]],
        strides=1, padding="same", activation=leaky_relu,
    )
    conv2 = spatial_attention(conv2)
    pool = tf.layers.max_pooling2d(conv2, [DCNN["pool"], DCNN["pool"]], strides=2)
    flat = tf.contrib.layers.flatten(pool)
    dense = tf.contrib.layers.fully_connected(flat, DCNN["feature_dim"], activation_fn=leaky_relu)
    dropped = tf.nn.dropout(dense, keep_prob)
    logits = tf.contrib.layers.fully_connected(dropped, 3, activation_fn=None)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_ph, logits=logits))
    train = tf.train.AdamOptimizer(DCNN["learning_rate"]).minimize(loss)
    return {
        "x": x_ph,
        "y": y_ph,
        "keep_prob": keep_prob,
        "dense": dense,
        "loss": loss,
        "train": train,
    }


def extract_features(images, labels, seed=DCNN["seed"]):
    """Fit the extractor and return the 200-dimensional features."""
    images = np.asarray(images, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    if images.ndim == 3:
        images = images.reshape(images.shape[0], -1)
    tf.set_random_seed(seed)
    tf.reset_default_graph()
    graph = build_graph()
    features = None
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        while True:
            _, step_loss = sess.run(
                [graph["train"], graph["loss"]],
                feed_dict={graph["x"]: images, graph["y"]: labels, graph["keep_prob"]: DCNN["keep_prob"]},
            )
            if DCNN["stop_loss"] > step_loss:
                break
            features = sess.run(
                graph["dense"],
                feed_dict={graph["x"]: images, graph["keep_prob"]: 1.0},
            )
    if features is None:
        raise RuntimeError("loss fell below the stop value before any feature export")
    return features


def save_features(path, features):
    pd.DataFrame(features).to_csv(path, index=False)

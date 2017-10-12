import math
import collections
import random
import numpy as np
import tensorflow as tf
import itertools
import time


def sum_rows(x):
    """Returns a vector summing up each row of the matrix x."""
    cols = tf.shape(x)[1]
    ones_shape = tf.stack([cols, 1])
    ones = tf.ones(ones_shape, x.dtype)
    return tf.reshape(tf.matmul(x, ones), [-1])


def compute_sampled_logits(weights,
                           biases,
                           labels,
                           inputs,
                           num_sampled,
                           num_classes,
                           num_true=1):
    if not isinstance(weights, list):
        weights = [weights]
    if labels.dtype != tf.int64:
        labels = tf.cast(labels, tf.int64)
    labels_flat = tf.reshape(labels, [-1])
    sampled_ids, true_expected_count, sampled_expected_count = tf.nn.log_uniform_candidate_sampler(
        true_classes=labels,
        num_true=num_true,
        num_sampled=num_sampled,
        unique=True,
        range_max=num_classes)

    true_w = tf.nn.embedding_lookup(weights, labels_flat)
    true_b = tf.nn.embedding_lookup(biases, labels_flat)
    sampled_w = tf.nn.embedding_lookup(weights, sampled_ids)
    sampled_b = tf.nn.embedding_lookup(biases, sampled_ids)
    dim = tf.shape(true_w)[1:2]
    new_true_w_shape = tf.concat([[-1, num_true], dim], 0)
    row_wise_dots = tf.multiply(tf.expand_dims(inputs, 1), tf.reshape(true_w, new_true_w_shape))
    dots_as_matrix = tf.reshape(row_wise_dots, tf.concat([[-1], dim], 0))
    true_logits = tf.reshape(sum_rows(dots_as_matrix), [-1, num_true])
    true_b = tf.reshape(true_b, [-1, num_true])
    true_logits += true_b
    sampled_b_vec = tf.reshape(sampled_b, [num_sampled])
    sampled_logits = tf.matmul(inputs, sampled_w, transpose_b=True) + sampled_b_vec

    return true_logits, sampled_logits


def nce_loss(weights,
             biases,
             labels,
             inputs,
             num_sampled,
             num_classes,
             num_true=1,
             v=None):
    batch_size = int(labels.get_shape()[0])
    if v is None:
        v = tf.ones([batch_size, 1])

    true_logits, sampled_logits = compute_sampled_logits(
        weights=weights,
        biases=biases,
        labels=labels,
        inputs=inputs,
        num_sampled=num_sampled,
        num_classes=num_classes,
        num_true=num_true)
    true_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(true_logits), logits=true_logits)
    sampled_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(sampled_logits), logits=sampled_logits)

    true_loss = tf.multiply(true_loss, v)
    return tf.div(tf.reduce_sum(true_loss) + tf.reduce_sum(sampled_loss), tf.constant(batch_size, dtype=tf.float32))

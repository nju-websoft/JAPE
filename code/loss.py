import tensorflow as tf
from params import *


def optimizer_loss(base_loss, norm_loss=None, base_loss_param=1, norm_loss_param=alpha):
    if norm_loss is None:
        loss = base_loss_param * base_loss
    else:
        loss = base_loss_param * base_loss + norm_loss_param * norm_loss
    optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss)
    return optimizer, loss


def only_pos_loss(phs, prs, pts):
    # base_loss = tf.reduce_sum(tf.reduce_sum(tf.abs(phs + prs - pts), 1))
    base_loss = tf.reduce_sum(tf.reduce_sum(tf.pow(phs + prs - pts, 2), 1))
    return optimizer_loss(base_loss)


def only_neg_loss(nhs, nrs, nts):
    neg_loss = tf.reduce_sum(tf.reduce_sum(tf.pow(nhs + nrs - nts, 2), 1))
    base_loss = - neg_param * neg_loss
    return optimizer_loss(base_loss)


def loss_with_neg(phs, prs, pts, nhs, nrs, nts):
    pos_loss = tf.reduce_sum(tf.reduce_sum(tf.pow(phs + prs - pts, 2), 1))
    neg_loss = tf.reduce_sum(tf.reduce_sum(tf.pow(nhs + nrs - nts, 2), 1))
    base_loss = pos_loss - neg_param * neg_loss
    return optimizer_loss(base_loss)


def sim_loss_sparse_with_kb12(ents_1, ents_2, cross_sim_mat, kb1_sim_mat, kb2_sim_mat):
    opt_vars = [v for v in tf.trainable_variables() if v.name.startswith("relation2vec")]
    trans_ents = tf.sparse_tensor_dense_matmul(cross_sim_mat, ents_2)
    trans_ents = tf.nn.l2_normalize(trans_ents, 1)
    base_loss = tf.reduce_sum(tf.reduce_sum(tf.pow(ents_1 - trans_ents, 2), 1))

    if inner_sim_param > 0.0:
        trans_kb1_ents = tf.sparse_tensor_dense_matmul(kb1_sim_mat, ents_1)
        trans_kb1_ents = tf.nn.l2_normalize(trans_kb1_ents, 1)
        base_loss += inner_sim_param * tf.reduce_sum(tf.reduce_sum(tf.pow(ents_1 - trans_kb1_ents, 2), 1))
        trans_kb2_ents = tf.sparse_tensor_dense_matmul(kb2_sim_mat, ents_2)
        trans_kb2_ents = tf.nn.l2_normalize(trans_kb2_ents, 1)
        base_loss += inner_sim_param * tf.reduce_sum(tf.reduce_sum(tf.pow(ents_2 - trans_kb2_ents, 2), 1))

    loss = sim_loss_param * base_loss
    optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss, var_list=opt_vars)
    return optimizer, loss







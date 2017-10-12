import random
import time
import copy

import multiprocessing
import numpy as np
import tensorflow as tf
from scipy import io
from sklearn import preprocessing

from params import *
from embed_utils import *
from triples_data import *


def generate_input(folder):
    triples1 = read_triples_ids(folder + 'triples_1')
    triples_data1 = Triples_Data(triples1)

    triples2 = read_triples_ids(folder + 'triples_2')
    triples_data2 = Triples_Data(triples2)

    ent_num = len(triples_data1.ents | triples_data2.ents)
    rel_num = len(triples_data1.rels | triples_data2.rels)
    triples_num = len(triples1) + len(triples2)
    print('all ents:', ent_num)
    print('all rels:', len(triples_data1.rels), len(triples_data2.rels), rel_num)
    print('all triples: %d + %d = %d' % (len(triples1), len(triples2), triples_num))

    refs1, refs2 = read_ref(folder + 'ref_ent_ids')
    ref1_list = copy.deepcopy(refs1)
    ref2_list = copy.deepcopy(refs2)

    print("To align:", len(refs2))
    sup_ents_pairs = read_pair_ids(folder + 'sup_ent_ids')

    return triples_data1, triples_data2, sup_ents_pairs, refs1, ref2_list, refs2, ref1_list, triples_num, ent_num, rel_num


def generate_pos_batch_of2KBs(triples_data1, triples_data2, step):
    # print(triples_data1.train_triples[0: 2])
    # print(triples_data2.train_triples[0: 2])
    assert batch_size % 2 == 0
    num1 = int(triples_data1.train_triples_num / (
        triples_data1.train_triples_num + triples_data2.train_triples_num) * batch_size)
    num2 = batch_size - num1
    start1 = step * num1
    start2 = step * num2
    end1 = start1 + num1
    end2 = start2 + num2
    if end1 > triples_data1.train_triples_num:
        end1 = triples_data1.train_triples_num
    if end2 > triples_data2.train_triples_num:
        end2 = triples_data2.train_triples_num
    pos_triples1 = triples_data1.train_triples[start1: end1]
    pos_triples2 = triples_data2.train_triples[start2: end2]
    return pos_triples1, pos_triples2


def generate_pos_batch(triples_data1, triples_data2, step):
    pos_triples1, pos_triples2 = generate_pos_batch_of2KBs(triples_data1, triples_data2, step)
    pos_triples1.extend(pos_triples2)
    assert len(pos_triples1) == batch_size
    return pos_triples1


def generate_pos_neg_batch(triples_data1, triples_data2, step, is_half=False, neg_scope=False, multi=1):
    pos_triples1, pos_triples2 = generate_pos_batch_of2KBs(triples_data1, triples_data2, step)

    if is_half:
        pos_triples11 = random.sample(pos_triples1, len(pos_triples1) // 2)
        pos_triples22 = random.sample(pos_triples2, len(pos_triples2) // 2)
        neg_triples1 = generate_neg_triples(pos_triples11, triples_data1, neg_scope)
        neg_triples2 = generate_neg_triples(pos_triples22, triples_data2, neg_scope)
    else:
        neg_triples1 = generate_neg_triples(pos_triples1, triples_data1, neg_scope)
        neg_triples2 = generate_neg_triples(pos_triples2, triples_data2, neg_scope)

    neg_triples1.extend(neg_triples2)
    if multi > 1:
        for i in range(multi - 1):
            neg_triples1.extend(generate_neg_triples(pos_triples1, triples_data1, neg_scope))
            neg_triples1.extend(generate_neg_triples(pos_triples2, triples_data2, neg_scope))

    pos_triples1.extend(pos_triples2)

    return pos_triples1, neg_triples1


def generate_neg_triples(pos_triples, triples_data, neg_scope):
    neg_triples = list()
    for (h, r, t) in pos_triples:
        h2, r2, t2 = h, r, t
        temp_scope, num = neg_scope, 0
        while True:
            choice = random.randint(0, 999)
            if choice < 500:
                if temp_scope:
                    h2 = random.sample(triples_data.r_hs_train[r], 1)[0]
                else:
                    h2 = random.sample(triples_data.ents_list, 1)[0]
                    # h2 = random.sample(triples_data.heads_list, 1)[0]
            elif choice >= 500:
                if temp_scope:
                    t2 = random.sample(triples_data.r_ts_train[r], 1)[0]
                else:
                    t2 = random.sample(triples_data.ents_list, 1)[0]
                    # t2 = random.sample(triples_data.tails_list, 1)[0]
            if not triples_data.exist(h2, r2, t2):
                break
            else:
                num += 1
                if num > 10:
                    temp_scope = False
        neg_triples.append((h2, r2, t2))
    return neg_triples


def get_all_sim_mat_sparse(folder):
    cross_sim_mat = preprocessing.normalize(io.mmread(folder + 'ents_sim.mtx'), norm='l1')
    kb1_sim_mat = preprocessing.normalize(io.mmread(folder + 'kb1_ents_sim.mtx'), norm='l1')
    kb2_sim_mat = preprocessing.normalize(io.mmread(folder + 'kb2_ents_sim.mtx'), norm='l1')
    return cross_sim_mat, kb1_sim_mat, kb2_sim_mat


def sparse_mat_2sparse_tensor(sparse_mat):
    print("sparse sim mat to sparse tensor")
    indices = list()
    values = list()
    shape = sparse_mat.shape
    for i in range(shape[0]):
        cols = sparse_mat.indices[sparse_mat.indptr[i]:sparse_mat.indptr[i + 1]]
        if len(cols) > 0:
            data = sparse_mat.data[sparse_mat.indptr[i]:sparse_mat.indptr[i + 1]]
            assert len(data) == len(cols)
            for j in range(len(data)):
                values.append(data[j])
                indices.append([i, cols[j]])
    return tf.SparseTensor(indices=indices, values=values, dense_shape=shape)


def get_ids_by_order(folder):
    ids_list1 = read_ents_by_order(folder + 'ent_ids_1')
    ids_list2 = read_ents_by_order(folder + 'ent_ids_2')
    return ids_list1, ids_list2


def valid(ent_embeddings, references_s, references_t_list, references_t, references_s_list, early_stop_flag1,
          early_stop_flag2, hits, top_k=[1, 5, 10, 50, 100], valid_threads=valid_mul):
    # if valid_threads > 1:
    #     res1, hits1 = valid_results_mul(ent_embeddings, references_s, references_t_list, 'X-EN:', top_k=top_k)
    #     res2, hits2 = valid_results_mul(ent_embeddings, references_t, references_s_list, 'EN-X:', top_k=top_k)
    # else:
    res1, hits1 = valid_results(ent_embeddings, references_s, references_t_list, 'X-EN:', top_k=top_k)
    res2, hits2 = valid_results(ent_embeddings, references_t, references_s_list, 'EN-X:', top_k=top_k)
    flag1 = early_stop_flag1 - res1
    flag2 = early_stop_flag2 - res2
    flag3 = hits1 - hits
    if flag1 < early_stop and flag2 < early_stop and flag3 < 0:
        print("early stop")
        return -1, -1, -1
    else:
        return res1, res2, hits1


def valid_results(embeddings, references_s, references_t, word, top_k=[1, 5, 10, 50, 100]):
    s_len = int(references_s.shape[0])
    t_len = int(references_t.shape[0])
    s_embeddings = tf.nn.embedding_lookup(embeddings, references_s)
    t_embeddings = tf.nn.embedding_lookup(embeddings, references_t)
    similarity_mat = tf.matmul(s_embeddings, t_embeddings, transpose_b=True)
    t = time.time()
    sim = similarity_mat.eval()
    num = [0 for k in top_k]
    mean = 0
    for i in range(s_len):
        ref = i
        rank = (-sim[i, :]).argsort()
        assert ref in rank
        rank_index = np.where(rank == ref)[0][0]
        mean += (rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                num[j] += 1
    acc = np.array(num) / s_len
    for i in range(len(acc)):
        acc[i] = round(acc[i], 4)
    mean /= s_len
    print("{} acc of top {} = {}, mean = {:.3f}, time = {:.3f} s ".
          format(word, top_k, acc, mean, time.time() - t))
    return mean / t_len, acc[2]


def cal_rank(task, sim, top_k):
    mean = 0
    num = [0 for k in top_k]
    for i in range(len(task)):
        ref = task[i]
        rank = (-sim[i, :]).argsort()
        assert ref in rank
        rank_index = np.where(rank == ref)[0][0]
        mean += (rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                num[j] += 1
    return mean, num


def valid_results_mul(embeddings, references_s, references_t, word, top_k=[1, 5, 10, 50, 100]):
    s_len = int(references_s.shape[0])
    t_len = int(references_t.shape[0])
    s_embeddings = tf.nn.embedding_lookup(embeddings, references_s)
    t_embeddings = tf.nn.embedding_lookup(embeddings, references_t)
    similarity_mat = tf.matmul(s_embeddings, t_embeddings, transpose_b=True)
    t = time.time()
    sim = similarity_mat.eval()
    total_num = [0 for k in top_k]
    total_mean = 0
    tasks = div_list(np.array(range(s_len)), 5)
    pool = multiprocessing.Pool(processes=len(tasks))
    reses = list()
    for task in tasks:
        reses.append(pool.apply_async(cal_rank, (task, sim[task, :], top_k)))
    pool.close()
    pool.join()
    for res in reses:
        mean, num = res.get()
        total_mean += mean
        total_num += np.array(num)
    acc = np.array(total_num) / s_len
    for i in range(len(acc)):
        acc[i] = round(acc[i], 4)
    total_mean /= s_len
    print("{} acc of top {} = {}, mean = {:.3f}, time = {:.3f} s ".
          format(word, top_k, acc, total_mean, time.time() - t))
    return total_mean / t_len, acc[2]


def random_unit_embeddings(dim1, dim2):
    return preprocessing.normalize(np.random.randn(dim1, dim2))


def generate_m(k):
    # mat = np.random.randint(0, k, size=[k, k])
    mat = np.random.randn(k, k)
    m = np.linalg.qr(mat)[0]
    return m


def valid_m(ent_embeddings, mat, references_s, references_t_list, early_stop_flag1, hits, top_k=[1, 5, 10, 50, 100]):
    res1, hits1 = valid_results_m(ent_embeddings, mat, references_s, references_t_list, 'X-EN:', top_k=top_k)
    flag1 = early_stop_flag1 - res1
    flag2 = hits1 - hits
    if flag1 <= early_stop and flag2 <= 0:
        print("early stop")
        exit()
    else:
        return res1, hits1


def valid_results_m(embeddings, mat, references_s, references_t, word, top_k=[1, 5, 10, 50, 100]):
    s_len = int(references_s.shape[0])
    t_len = int(references_t.shape[0])
    s_embeddings = tf.nn.embedding_lookup(embeddings, references_s)
    s_embeddings = tf.matmul(s_embeddings, mat)
    t_embeddings = tf.nn.embedding_lookup(embeddings, references_t)
    similarity_mat = tf.matmul(s_embeddings, t_embeddings, transpose_b=True)
    t = time.time()
    sim = similarity_mat.eval()
    num = [0 for k in top_k]
    mean = 0
    for i in range(s_len):
        ref = i
        rank = (-sim[i, :]).argsort()
        assert ref in rank
        rank_index = np.where(rank == ref)[0][0]
        mean += (rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                num[j] += 1
    acc = np.array(num) / s_len
    for i in range(len(acc)):
        acc[i] = round(acc[i], 4)
    mean /= s_len
    print("{} acc of top {} = {}, mean = {:.3f}, time = {:.3f} s ".
          format(word, top_k, acc, mean, time.time() - t))
    return mean / t_len, acc[2]


def save_embeddings(folder, ent_embeddings, rel_embeddings, references_s, references_t_list):
    print("save embeddings...")
    final_ent_embeddings = ent_embeddings.eval()
    final_rel_embeddings = rel_embeddings.eval()
    ref1_ents_embeddings = tf.nn.embedding_lookup(ent_embeddings, references_s).eval()
    ref2_ents_embeddings = tf.nn.embedding_lookup(ent_embeddings, references_t_list).eval()
    np.save(folder + "ents_vec", final_ent_embeddings)
    np.save(folder + "rels_vec", final_rel_embeddings)
    np.save(folder + "ref1_vec", ref1_ents_embeddings)
    np.save(folder + "ref2_vec", ref2_ents_embeddings)

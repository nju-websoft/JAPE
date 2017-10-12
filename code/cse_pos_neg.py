import math
import sys
from loss import *
from embed_func import *


def structure_embedding(folder):
    triples_data1, triples_data2, sup_ents_pairs, ref_s, ref_t_list, ref_t, ref_s_list, triples_num, ent_num, rel_num = generate_input(
        folder)
    cross_sim_mat, kb1_sim_mat, kb2_sim_mat = get_all_sim_mat_sparse(folder)
    ids_list1, ids_list2 = get_ids_by_order(folder)

    graph = tf.Graph()
    with graph.as_default():
        pos_hs = tf.placeholder(tf.int32, shape=[None])
        pos_rs = tf.placeholder(tf.int32, shape=[None])
        pos_ts = tf.placeholder(tf.int32, shape=[None])
        neg_hs = tf.placeholder(tf.int32, shape=[None])
        neg_rs = tf.placeholder(tf.int32, shape=[None])
        neg_ts = tf.placeholder(tf.int32, shape=[None])
        flag = tf.placeholder(tf.bool)

        with tf.variable_scope('relation2vec' + 'embedding'):
            ent_embeddings = tf.Variable(tf.truncated_normal([ent_num, embed_size], stddev=1.0 / math.sqrt(embed_size)))
            rel_embeddings = tf.Variable(tf.truncated_normal([rel_num, embed_size], stddev=1.0 / math.sqrt(embed_size)))
            ent_embeddings = tf.nn.l2_normalize(ent_embeddings, 1)
            rel_embeddings = tf.nn.l2_normalize(rel_embeddings, 1)
            references_s = tf.constant(ref_s, dtype=tf.int32)
            references_t_list = tf.constant(ref_t_list, dtype=tf.int32)
            references_t = tf.constant(ref_t, dtype=tf.int32)
            references_s_list = tf.constant(ref_s_list, dtype=tf.int32)

        with tf.variable_scope('sparse' + 'sim'):
            cross_sparse_sim = sparse_mat_2sparse_tensor(cross_sim_mat)
            kb1_sparse_sim = sparse_mat_2sparse_tensor(kb1_sim_mat)
            kb2_sparse_sim = sparse_mat_2sparse_tensor(kb2_sim_mat)

        ents_1 = tf.nn.embedding_lookup(ent_embeddings, ids_list1)
        ents_2 = tf.nn.embedding_lookup(ent_embeddings, ids_list2)
        phs = tf.nn.embedding_lookup(ent_embeddings, pos_hs)
        prs = tf.nn.embedding_lookup(rel_embeddings, pos_rs)
        pts = tf.nn.embedding_lookup(ent_embeddings, pos_ts)
        nhs = tf.nn.embedding_lookup(ent_embeddings, neg_hs)
        nrs = tf.nn.embedding_lookup(rel_embeddings, neg_rs)
        nts = tf.nn.embedding_lookup(ent_embeddings, neg_ts)

        rel_optimizer, rel_loss = tf.cond(flag, lambda: only_pos_loss(phs, prs, pts),
                                          lambda: only_neg_loss(nhs, nrs, nts))
        sim_optimizer, sim_loss = sim_loss_sparse_with_kb12(ents_1, ents_2, cross_sparse_sim, kb1_sparse_sim,
                                                            kb2_sparse_sim)
        total_start_time = time.time()
        early_stop_flag1, early_stop_flag2, hits = 1, 1, 0

        with tf.Session(graph=graph) as sess:
            tf.global_variables_initializer().run()
            num_steps = triples_num // batch_size
            for epoch in range(num_epochs):
                train_loss = 0
                start = time.time()

                if epoch % 2 == 0:
                    for step in range(num_steps):
                        batch_pos, batch_neg = generate_pos_neg_batch(triples_data1, triples_data2, step)
                        for i in range(2):
                            train_flag = True if i % 2 == 0 else False
                            feed_dict = {pos_hs: [x[0] for x in batch_pos],
                                         pos_rs: [x[1] for x in batch_pos],
                                         pos_ts: [x[2] for x in batch_pos],
                                         neg_hs: [x[0] for x in batch_neg],
                                         neg_rs: [x[1] for x in batch_neg],
                                         neg_ts: [x[2] for x in batch_neg],
                                         flag: train_flag}
                            (_, loss_val) = sess.run([rel_optimizer, rel_loss], feed_dict=feed_dict)
                            train_loss += loss_val
                else:
                    batch_pos, batch_neg = generate_pos_neg_batch(triples_data1, triples_data2, 1)
                    feed_dict = {pos_hs: [x[0] for x in batch_pos],
                                 pos_rs: [x[1] for x in batch_pos],
                                 pos_ts: [x[2] for x in batch_pos],
                                 neg_hs: [x[0] for x in batch_neg],
                                 neg_rs: [x[1] for x in batch_neg],
                                 neg_ts: [x[2] for x in batch_neg],
                                 flag: True}
                    (_, loss_val) = sess.run([sim_optimizer, sim_loss], feed_dict=feed_dict)
                    train_loss += loss_val
                random.shuffle(triples_data1.train_triples)
                random.shuffle(triples_data2.train_triples)
                end = time.time()
                loss_print = "rel loss" if epoch % 2 == 0 else "sim loss"
                print("{}/{}, {} = {:.3f}, time = {:.3f} s".format(epoch, num_epochs, loss_print, train_loss,
                                                                   end - start))
                if (epoch % print_validation == 0 or epoch == num_epochs - 1) and epoch >= 300:
                    # if epoch % print_validation == 0 or epoch == num_epochs - 1:
                    early_stop_flag1, early_stop_flag2, hits = valid(ent_embeddings, references_s, references_t_list,
                                                                     references_t, references_s_list, early_stop_flag1,
                                                                     early_stop_flag2, hits)
                    if early_stop_flag1 < 0 and early_stop_flag2 < 0 and hits < 0:
                        save_embeddings(folder, ent_embeddings, rel_embeddings, references_s, references_t_list)
                        print_time(time.time() - total_start_time)
                        exit()


if __name__ == '__main__':
    assert len(sys.argv) == 3
    data_folder = sys.argv[1]
    supervised_ent_rel_ratio = sys.argv[2]
    folder = radio_2file(supervised_ent_rel_ratio, data_folder)
    print("neg param", neg_param, "; split", "; sim param", sim_loss_param, "; inner sim param", inner_sim_param)
    structure_embedding(folder)

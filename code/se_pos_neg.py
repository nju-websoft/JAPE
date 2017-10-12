import math
import sys
from loss import *
from embed_func import *


def structure_embedding(folder):
    triples_data1, triples_data2, sup_ents_pairs, ref_s, ref_t_list, ref_t, ref_s_list, triples_num, ent_num, rel_num = generate_input(
        folder)

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

        phs = tf.nn.embedding_lookup(ent_embeddings, pos_hs)
        prs = tf.nn.embedding_lookup(rel_embeddings, pos_rs)
        pts = tf.nn.embedding_lookup(ent_embeddings, pos_ts)
        nhs = tf.nn.embedding_lookup(ent_embeddings, neg_hs)
        nrs = tf.nn.embedding_lookup(rel_embeddings, neg_rs)
        nts = tf.nn.embedding_lookup(ent_embeddings, neg_ts)
        optimizer, loss = tf.cond(flag, lambda: only_pos_loss(phs, prs, pts), lambda: only_neg_loss(nhs, nrs, nts))

        total_start_time = time.time()
        early_stop_flag1, early_stop_flag2, hits = 1, 1, 0

        with tf.Session(graph=graph) as sess:
            tf.global_variables_initializer().run()
            num_steps = triples_num // batch_size
            for epoch in range(num_epochs):
                pos_loss = 0
                start = time.time()
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
                        (_, loss_val) = sess.run([optimizer, loss], feed_dict=feed_dict)
                        pos_loss += loss_val
                random.shuffle(triples_data1.train_triples)
                random.shuffle(triples_data2.train_triples)
                end = time.time()
                print("{}/{}, relation_loss = {:.3f}, time = {:.3f} s".format(epoch, num_epochs, pos_loss, end - start))
                # if (epoch % print_validation == 0 or epoch == num_epochs - 1) and epoch >= 200:
                if epoch % print_validation == 0 or epoch == num_epochs - 1:
                    early_stop_flag1, early_stop_flag2, hits = valid(ent_embeddings, references_s, references_t_list,
                                                                     references_t, references_s_list, early_stop_flag1,
                                                                     early_stop_flag2, hits)
                    if early_stop_flag1 < 0 and early_stop_flag2 < 0 and hits < 0:
                        print_time(time.time() - total_start_time)
                        exit()

if __name__ == '__main__':
    assert len(sys.argv) == 3
    data_folder = sys.argv[1]
    supervised_ent_rel_ratio = sys.argv[2]
    folder = radio_2file(supervised_ent_rel_ratio, data_folder)
    print("neg param", neg_param, "split")
    structure_embedding(folder)

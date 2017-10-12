import sys

from data_utils import *
from conf import *
from attr_data_methods import *
from attr2vec_func import *

data_frequent_p = 0.95
batch_size = 2000
num_sampled_negs = 1000
num_train = 100
min_frequency = 5  # small data 5
min_props = 2
lr = 0.1
v = 2

embedding_size = 100
valid_size = 16
valid_window = 100
valid_examples = np.random.choice(valid_window, valid_size, replace=False)


def get_common(props_list, props_set):
    print("total props:", len(props_set))
    print("total prop frequency:", len(props_list))
    n = int(data_frequent_p * len(props_set))
    most_frequent_props = collections.Counter(props_list).most_common(n)
    print(most_frequent_props[0:10])
    most_frequent_props = most_frequent_props[len(props_set) - n:]
    common_props_ids = dict()
    for prop, freq in most_frequent_props:
        if freq >= min_frequency and prop not in common_props_ids:
            common_props_ids[prop] = len(common_props_ids)
    return common_props_ids


def load_data(attr_folder, rel_train_data_folder, attrs_range_file, en_attrs_range_file):
    sup_ents_pairs = read_pair_ids(rel_train_data_folder + 'sup_ent_ids')
    kb1_ids, _, _, _ = read_ids(rel_train_data_folder + 'ent_ids_1')
    kb2_ids, _, _, _ = read_ids(rel_train_data_folder + 'ent_ids_2')
    sup_ents_dict = pair_2dict(sup_ents_pairs)
    props_set = set()
    props_list = []
    data = []

    attrs1 = read_attrs(attr_folder + 'training_attrs_1')
    attrs2 = read_attrs(attr_folder + 'training_attrs_2')
    attrs1 = merge_dicts(attrs1, attrs2)

    for uri in attrs1.keys():
        props_list.extend(list(attrs1.get(uri)))
        props_set |= attrs1.get(uri)

    common_props_ids = get_common(props_list, props_set)
    reverse_props_ids = dict(zip(common_props_ids.values(), common_props_ids.keys()))

    attrs_with_ids = dict()
    prop_totals = []
    for uri in attrs1.keys():
        props = attrs1.get(uri)
        props_idxs = []
        for p in props:
            if p in common_props_ids:
                index_p = common_props_ids[p]
                props_idxs.append(index_p)
        if len(props_idxs) >= min_props:
            prop_totals.append(len(props_idxs))
            attrs_with_ids[uri] = set(props_idxs)
            for p_id, context_p in itertools.combinations(props_idxs, 2):
                if p_id != context_p:
                    data.append((p_id, context_p))

    for ent in sup_ents_dict:
        kb2_ent = sup_ents_dict.get(ent)
        ent_uri = kb1_ids[ent]
        kb2_ent_uri = kb2_ids[kb2_ent]
        if ent_uri in attrs_with_ids and kb2_ent_uri in attrs_with_ids:
            ent_props = attrs_with_ids.get(ent_uri)
            ent2_props = attrs_with_ids.get(kb2_ent_uri)
            for p_id, context_p in itertools.product(ent_props, ent2_props):
                if p_id != context_p:
                    for i in range(10):
                        data.append((p_id, context_p))
                        data.append((context_p, p_id))

    num_steps = num_train * int(len(data) / batch_size)

    range_dict = read_attrs_range(attrs_range_file)
    en_range_dict = read_attrs_range(en_attrs_range_file)
    range_vec = list()
    for i in range(len(common_props_ids)):
        assert i in reverse_props_ids
        attr_uri = reverse_props_ids[i]
        if attr_uri in range_dict:
            range_vec.append(range_dict.get(attr_uri))
        elif attr_uri in en_range_dict:
            range_vec.append(en_range_dict.get(attr_uri))
        else:
            range_vec.append(0)

    return data, len(common_props_ids), common_props_ids, reverse_props_ids, num_steps, range_vec


def get_range_weight(range_vec, id1, id2):
    if range_vec[id1] == range_vec[id2]:
        return v
    return 1.0


def generate_batch_random(data_list, batch_size, range_vec):
    batch_data = random.sample(data_list, batch_size)
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    range_type = np.ndarray(shape=(batch_size, 1), dtype=np.float32)
    for i in range(len(batch_data)):
        batch[i] = batch_data[i][0]
        labels[i, 0] = batch_data[i][1]
        range_type[i, 0] = get_range_weight(range_vec, batch[i], labels[i, 0])
    return batch, labels, range_type


def dict2file(dictionary, embeddings, meta_out_file):
    fw = open(meta_out_file, 'w', encoding='utf8')
    for i in range(embeddings.shape[0]):
        fw.write(dictionary[i] + "\n")
    fw.close()


def embedding2file(embeddings, embeddings_out_file):
    print("Embedding:", embeddings.shape)
    fw = open(embeddings_out_file, 'w', encoding='utf8')
    for i in range(embeddings.shape[0]):
        line = ''
        for j in range(embeddings.shape[1]):
            line = line + str(embeddings[i, j]) + '\t'
        fw.write(line.strip() + '\n')
    fw.close()


def learn_vec(attr_folder, rel_train_data_folder, attrs_range_file, en_attrs_range_file):
    init_width = 1.0 / math.sqrt(embedding_size)
    prop_vec_file = rel_train_data_folder + 'attrs_vec'
    prop_embeddings_file = rel_train_data_folder + 'attrs_embeddings'
    meta_out_file = rel_train_data_folder + 'attrs_meta'
    data, props_size, dictionary, reverse_dictionary, num_steps, range_vecss = load_data(attr_folder,
                                                                                         rel_train_data_folder,
                                                                                         attrs_range_file,
                                                                                         en_attrs_range_file)
    # if num_steps < 50000:
    #     num_steps = 50000
    print("number of steps:", num_steps)

    graph = tf.Graph()
    with graph.as_default():
        # 输入变量
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        range_vecs = tf.placeholder(tf.float32, shape=[batch_size, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        with tf.variable_scope('props' + 'embedding'):
            embeddings = tf.Variable(tf.random_uniform([props_size, embedding_size], -init_width, init_width))
            embeddings = tf.nn.l2_normalize(embeddings, 1)
            nce_weights = tf.Variable(tf.truncated_normal([props_size, embedding_size], stddev=init_width))
            # nce_weights = tf.Variable(tf.zeros([props_size, embedding_size]))
            nce_biases = tf.Variable(tf.zeros([props_size]))

        embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        loss = nce_loss(nce_weights, nce_biases, train_labels, embed, num_sampled_negs, props_size, v=range_vecs)
        optimizer = tf.train.AdagradOptimizer(lr).minimize(loss)

        init = tf.global_variables_initializer()

    with tf.Session(graph=graph) as session:
        init.run()
        average_loss = 0
        t = time.time()
        for step in range(num_steps):
            batch_inputs, batch_labels, range_types = generate_batch_random(data, batch_size, range_vecss)
            feed_dict = {train_inputs: batch_inputs,
                         train_labels: batch_labels,
                         range_vecs: range_types}
            _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += loss_val
            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                print("average loss at step", step, ":", average_loss, "time = ", round(time.time() - t, 2))
                t = time.time()
                average_loss = 0

            if step % 10000 == 0:
                final_embeddings = embeddings.eval()
                np.save(prop_vec_file, final_embeddings)
                embedding2file(final_embeddings, prop_embeddings_file)
                dict2file(reverse_dictionary, final_embeddings, meta_out_file)
        final_embeddings = embeddings.eval()
        np.save(prop_vec_file, final_embeddings)
        embedding2file(final_embeddings, prop_embeddings_file)
        dict2file(reverse_dictionary, final_embeddings, meta_out_file)


if __name__ == '__main__':
    assert len(sys.argv) == 5
    attr_folder = sys.argv[1]
    rel_train_data_folder = sys.argv[2]
    attr_range_file = sys.argv[3]
    en_attr_range_file = sys.argv[4]
    learn_vec(attr_folder, rel_train_data_folder, attr_range_file, en_attr_range_file)
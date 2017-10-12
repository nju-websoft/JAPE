import os


def read_ids(ids_file):
    file = open(ids_file, 'r', encoding='utf8')
    ids_list = list()
    for line in file.readlines():
        params = line.strip('\n').split('\t')
        assert len(params) == 2
        id = int(params[0])
        ids_list.append(id)
    return ids_list


def embedding2file(embeddings, embeddings_out_file):
    print("Embedding:", embeddings.shape)
    fw = open(embeddings_out_file, 'w', encoding='utf8')
    for i in range(embeddings.shape[0]):
        line = ''
        for j in range(embeddings.shape[1]):
            line = line + str(embeddings[i, j]) + '\t'
        fw.write(line.strip() + '\n')
    fw.close()


def print_time(t):
    print('time:{:.3f} s'.format(t))


def radio_2file(radio, folder):
    path = folder + str(radio).replace('.', '_')
    if not os.path.exists(path):
        os.makedirs(path)
    return path + '/'


def read_pairs(file_path):
    file = open(file_path, 'r', encoding='utf8')
    pairs = list()
    for line in file.readlines():
        params = line.strip('\n').split('\t')
        assert len(params) == 2
        pairs.append((params[0], params[1]))
    file.close()
    return pairs


def read_pair_ids(file_path):
    file = open(file_path, 'r', encoding='utf8')
    pairs = list()
    for line in file.readlines():
        params = line.strip('\n').split('\t')
        assert len(params) == 2
        pairs.append((int(params[0]), int(params[1])))
    file.close()
    return pairs


def pair_2set(pairs):
    s1, s2 = set(), set()
    for pair in pairs:
        s1.add(pair[0])
        s2.add(pair[1])
    return s1, s2


def read_triples_ids(file_path):
    triples = list()
    file = open(file_path, 'r', encoding='utf8')
    for line in file.readlines():
        params = line.strip('\n').split('\t')
        assert len(params) == 3
        h = int(params[0])
        r = int(params[1])
        t = int(params[2])
        triples.append((h, r, t))
    return triples


def read_ref(file_path):
    refs = list()
    reft = list()
    file = open(file_path, 'r', encoding='utf8')
    for line in file.readlines():
        params = line.strip('\n').split('\t')
        assert len(params) == 2
        e1 = int(params[0])
        e2 = int(params[1])
        refs.append(e1)
        reft.append(e2)
    assert len(refs) == len(reft)
    return refs, reft


def read_ents_by_order(ids_file):
    file = open(ids_file, 'r', encoding='utf8')
    ids_list = list()
    for line in file.readlines():
        params = line.strip('\n').split('\t')
        assert len(params) == 2
        ids_list.append(int(params[0]))
    return ids_list


def pair_2_rev_dict(pairs):
    d = dict()
    for pair in pairs:
        if pair[1] not in d:
            d[pair[1]] = pair[0]
        else:
            print("Error")
    return d


def pair_2int_set(pairs):
    s1, s2 = set(), set()
    for pair in pairs:
        s1.add(int(pair[0]))
        s2.add(int(pair[1]))
    return s1, s2


def div_list(ls, n):
    ls_len = len(ls)
    if n <= 0 or 0 == ls_len:
        return []
    if n > ls_len:
        return []
    elif n == ls_len:
        return [[i] for i in ls]
    else:
        j = ls_len // n
        k = ls_len % n
        ls_return = []
        for i in range(0, (n - 1) * j, j):
            ls_return.append(ls[i:i + j])
        ls_return.append(ls[(n - 1) * j:])
        return ls_return

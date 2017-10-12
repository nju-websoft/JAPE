import os


def parse_attr_ttl_lines(line):
    if "> " not in line:
        return None
    params = line.strip().strip('\n').strip('.').strip().split('> ')
    if ".dbpedia.org/resource/" in params[2]:
        return None
    if len(params) > 3:
        print(line)
        print(params)
    assert len(params) == 3
    ent = params[0].strip().lstrip('<').rstrip('>').strip()
    attr = params[1].strip().lstrip('<').rstrip('>').strip()
    literal = params[2].strip().strip('.').lstrip('<').rstrip('>').strip()
    return ent, attr, literal


def parse_ttl_lines(line):
    params = line.strip().strip('\n').split(' ')[0:3]
    ent_h = params[0].lstrip('<').rstrip('>').strip()
    prop = params[1].lstrip('<').rstrip('>').strip()
    ent_t = params[2].lstrip('<').rstrip('>').strip()
    return ent_h, prop, ent_t


def attrs_2file(attrs, file_path):
    file = open(file_path, 'w', encoding='utf8')
    for k in attrs.keys():
        atts = attrs.get(k)
        line = k
        for att in atts:
            line += ('\t' + att)
        file.write(line + '\n')
    file.close()


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


def read_interlink(ill_ttl, ent_t_prefix='http://dbpedia.org/resource/'):
    file = open(ill_ttl, 'r', encoding='utf8')
    ill_pairs = list()
    head_set, tail_set = set(), set()
    for line in file.readlines():
        ent1, _, ent2 = parse_ttl_lines(line)
        # 保证一对一的匹配
        if ent2.startswith(ent_t_prefix) and ent1 not in head_set and ent2 not in tail_set:
            ill_pairs.append((ent1, ent2))
            head_set.add(ent1)
            tail_set.add(ent2)
    return ill_pairs, head_set, tail_set


def read_attrs(attrs_file):

    attrs_dic = dict()
    with open(attrs_file, 'r', encoding='utf8') as file:
        for line in file:
            params = line.strip().strip('\n').split('\t')
            if len(params) >= 2:
                attrs_dic[params[0]] = set(params[1:])
            else:
                print(line)
    return attrs_dic


def print_line(*line):
    print()
    if len(line) == 0:
        print("====================================")
    else:
        for i in line:
            print(i, end=" ")
    print()


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


def pairs_2file(pairs, file_path):
    file = open(file_path, 'w', encoding='utf8')
    for pair in pairs:
        file.write(str(pair[0]) + '\t' + str(pair[1]) + '\n')
    file.close()


def pairs_ids_2file(pairs, dic1, dic2, file_path):
    file = open(file_path, 'w', encoding='utf8')
    for pair in pairs:
        assert pair[0] in dic1 and pair[1] in dic2
        file.write(str(dic1[pair[0]]) + '\t' + str(dic2[pair[1]]) + '\n')
    file.close()


def read_ttl_triples(ttl_file_path):
    ttl_file = open(ttl_file_path, 'r', encoding='utf8')
    triples = list()
    for line in ttl_file.readlines():
        ent_h, prop, ent_t = parse_ttl_lines(line)
        triples.append((ent_h, prop, ent_t))
    ttl_file.close()
    return triples


def read_triples(triples_file_path):
    if triples_file_path is None:
        return set()
    file = open(triples_file_path, 'r', encoding='utf8')
    triples = set()
    for line in file.readlines():
        ent_h, prop, ent_t = line.strip('\n').split('\t')
        triples.add((ent_h, prop, ent_t))
    file.close()
    return triples


def read_triple_ids(triples_file_path):
    if triples_file_path is None:
        return set()
    file = open(triples_file_path, 'r', encoding='utf8')
    triples = set()
    for line in file.readlines():
        ent_h, prop, ent_t = line.strip('\n').split('\t')
        triples.add((int(ent_h), int(prop), int(ent_t)))
    file.close()
    return triples


def triples_2file(triples, file_path):
    file = open(file_path, 'w', encoding='utf8')
    for triple in triples:
        file.write(triple[0] + '\t' + triple[1] + '\t' + triple[2] + '\n')
    file.close()


def pair_2dict(pairs):
    d = dict()
    for pair in pairs:
        if pair[0] not in d:
            d[pair[0]] = pair[1]
        else:
            print("Error")
    return d


def pair_2dict_rev(pairs):
    d = dict()
    for pair in pairs:
        if pair[1] not in d:
            d[pair[1]] = pair[0]
        else:
            print("Error")
    return d


def pair_2set(pairs):
    s1, s2 = set(), set()
    for pair in pairs:
        s1.add(pair[0])
        s2.add(pair[1])
    return s1, s2


def parse_triples(triples):
    ents, rels = set(), set()
    for triple in triples:
        ents.add(triple[0])
        rels.add(triple[1])
        ents.add(triple[2])
    return ents, rels


def parse_triples_heads(triples):
    heads, rels, tails = set(), set(), set()
    for triple in triples:
        heads.add(triple[0])
        rels.add(triple[1])
        tails.add(triple[2])
    return heads, rels, tails


def ids_2file(ids_mapping, path):
    ids_mapping = sorted(ids_mapping.items(), key=lambda d: d[1])
    fw = open(path, 'w', encoding='utf8')
    max = -1
    for uri, id in ids_mapping:
        if id > max:
            max = id
        fw.write(str(id) + '\t' + uri + '\n')
    print("max id:", max)
    fw.close()


def radio_2file(radio, folder):
    path = folder + str(radio).replace('.', '_')
    if not os.path.exists(path):
        os.makedirs(path)
    return path + '/'


def is_suffix_equal(uri1, uri2):
    uri1_params = uri1.split('/')
    uri2_params = uri2.split('/')
    uri1_suffix = uri1_params[-1]
    uri2_suffix = uri2_params[-1]
    return len(uri1_params) == len(uri2_params) and uri1_suffix == uri2_suffix


def merge_dicts(dict1, dict2):
    for k in dict1.keys():
        vs = dict1.get(k)
        dict2[k] = dict2.get(k, set()) | vs
    return dict2


def add_dict_kv(dic, k, v):
    vs = dic.get(k, set())
    vs.add(v)
    dic[k] = vs


def add_dict_one(dic, k):
    v = dic.get(k, 0)
    v += 1
    dic[k] = v


def add_dict_kvs(dic, k, vset):
    vs = dic.get(k, set())
    vs = vs | vset
    dic[k] = vs


def sup_attrs_2file(attrs, file_path):
    file = open(file_path, 'w', encoding='utf8')
    for k in attrs.keys():
        attr = attrs.get(k)
        file.write(k + '\t' + attr + '\n')
    file.close()


def read_ids(ids_file):
    file = open(ids_file, 'r', encoding='utf8')
    dic, reversed_dic, ids_set, uris_set = dict(), dict(), set(), set()
    for line in file.readlines():
        params = line.strip('\n').split('\t')
        assert len(params) == 2
        id = int(params[0])
        uri = params[1]
        dic[id] = uri
        reversed_dic[uri] = id
        ids_set.add(id)
        uris_set.add(uri)
    assert len(dic) == len(reversed_dic)
    assert len(ids_set) == len(uris_set)
    return dic, reversed_dic, ids_set, uris_set


def read_ents_by_order(ids_file):
    file = open(ids_file, 'r', encoding='utf8')
    uri_list = list()
    ids_uris_dict = dict()
    uris_ids_dict = dict()
    for line in file.readlines():
        params = line.strip('\n').split('\t')
        assert len(params) == 2
        uri_list.append(params[1])
        ids_uris_dict[int(params[0])] = params[1]
        uris_ids_dict[params[1]] = int(params[0])
    return uri_list, ids_uris_dict, uris_ids_dict


def read_lines(file_path):
    if file_path is None:
        return []
    file = open(file_path, 'r', encoding='utf8')
    return file.readlines()


if __name__ == '__main__':
    line = "<http://ja.dbpedia.org/resource/Dungeons_and_Dragons> <http://www.w3.org/2000/01/rdf-schema#label> \"Dungeons and Dragons\"@ja ."
    print(parse_attr_ttl_lines(line))

from conf import *
from data_utils import *


def generate_all_attr_data(folder, lines):
    print("generate all attrs...all attr lines", len(lines))
    attrs = dict()
    attr_triples = set()
    line_file = open(folder + ALL_ATTR_TRIPLES_FILE, 'w', encoding='utf8')
    for line in lines:
        triple = parse_ttl_lines(line)
        if is_attributed_triple(triple):
            add_dict_kv(attrs, triple[0], triple[1])
            line_file.write(line)
            attr_triples.add(triple)
    print("num of ents has attrs:", len(attrs))
    print("num of attr triples", len(attr_triples))
    line_file.close()
    attrs_2file(attrs, folder + ALL_ATTRS_FILE)
    del attrs
    dic = handle_attrs_range(attr_triples)
    file = open(folder + ATTR_RANGE_FILE, 'w', encoding='utf8')
    for k in dic.keys():
        v = dic.get(k)
        file.write(k + '\t' + str(v) + '\n')
    file.close()


def is_attributed_triple(triple):
    if triple[2].startswith('http://'):
        return False
    if len(triple[1].split('/')[-1]) < 2:
        return False
    return True


def get_type(value):
    value = value.strip()
    # if value.endswith("@en") or value.endswith("@de") or value.endswith("@fr") or value.endswith("@zh"):
    #     return 0  # "String"
    if value.endswith("^^<http://www.w3.org/2001/XMLSchema#integer"):
        return 1  # "Integer"
    if value.endswith("^^<http://www.w3.org/2001/XMLSchema#double"):
        return 2  # "Double"
    if value.endswith("^^<http://www.w3.org/2001/XMLSchema#date"):
        return 3  # "Date"
    return 0  # "String"


def handle_attrs_range(triples):
    range_type_dict = dict()
    # datetype = set()
    for triple in triples:
        t = get_type(triple[2])
        # if '^^' in triple[2]:
        #     params = triple[2].split('^^')
        #     datetype.add(params[-1])
        first_dict = range_type_dict.get(triple[1], dict())
        add_dict_one(first_dict, t)
        range_type_dict[triple[1]] = first_dict
    # for t in datetype:
    #     print(t)
    range_type_dict_final = dict()
    for k in range_type_dict.keys():
        dic = range_type_dict.get(k)
        sorts = sorted(dic.items(), key=lambda x: x[1], reverse=True)
        t = sorts[0][0]
        range_type_dict_final[k] = t
    return range_type_dict_final


################################# 我是分割线 #################################

def generate_all_attrs(attrs_file, ib_props_ttl, ib_props_mapped_ttl, mb_literals_ttl):
    attrs1 = read_ttl_2attrs_dict(ib_props_ttl)
    attrs2 = read_ttl_2attrs_dict(ib_props_mapped_ttl)
    attrs3 = read_ttl_2attrs_dict(mb_literals_ttl)
    attrs = merge_dicts(merge_dicts(attrs1, attrs2), attrs3)
    print("total num of ents has attrs:", len(attrs))
    attrs_2file(attrs, attrs_file)


def read_ttl_2attrs_dict(ttl_file_path):
    if ttl_file_path is None:
        return dict()
    attrs = dict()
    file = open(ttl_file_path, 'r', encoding='utf8')
    for line in file.readlines():
        triple = parse_ttl_lines(line)
        if is_attributed_triple(triple):
            add_dict_kv(attrs, triple[0], triple[1])
    print("num of ents has attrs:", len(attrs))
    return attrs


def generate_matched_attrs(kb1_attr, kb2_attrs_set):
    if kb1_attr in kb2_attrs_set:
        return kb1_attr
    else:
        for rel in kb2_attrs_set:
            if is_suffix_equal(rel, kb1_attr):
                return rel
    return None


def generate_sup_attrs(attrs_set1, attrs_set2):
    sup_attrs_dict = dict()
    for attr in attrs_set1:
        ref = generate_matched_attrs(attr, attrs_set2)
        if ref is not None and ref not in sup_attrs_dict:
            sup_attrs_dict[attr] = ref
    print_line("matched attrs:", len(sup_attrs_dict))
    return sup_attrs_dict


def replace_attrs_by_sups(kb1_attrs_dict, sup_attrs_dict):
    attrs_dict = dict()
    for uri in kb1_attrs_dict.keys():
        attrs = kb1_attrs_dict.get(uri)
        for attr in attrs:
            new_attrs = sup_attrs_dict.get(attr) if attr in sup_attrs_dict else attr
            add_dict_kv(attrs_dict, uri, new_attrs)
    return attrs_dict


def generate_attrs_train_data(data_folder, s_attrs_file, t_attrs_file, is_sup_attrs):
    kb1_triples = read_triples(data_folder + S_TRIPLES)
    kb2_triples = read_triples(data_folder + T_TRIPLES)
    kb1_ents_set, _, _ = parse_triples_heads(kb1_triples)
    kb2_ents_set, _, _ = parse_triples_heads(kb2_triples)
    kb1_attrs_dict, attrs_set1 = get_attrs(read_attrs(s_attrs_file), kb1_ents_set)
    print("num of attrs1:", len(attrs_set1))
    kb2_attrs_dict, attrs_set2 = get_attrs(read_attrs(t_attrs_file), kb2_ents_set)
    print("num of attrs2:", len(attrs_set2))
    if is_sup_attrs:
        sup_attrs_dict = generate_sup_attrs(attrs_set1, attrs_set2)
        # 这里是替换
        kb1_attrs_dict = replace_attrs_by_sups(kb1_attrs_dict, sup_attrs_dict)
        sup_attrs_2file(sup_attrs_dict, data_folder + SUP_ATTRS_FILE)

    attrs_2file(kb1_attrs_dict, data_folder + TRAINING_ATTRS_FILE1)
    attrs_2file(kb2_attrs_dict, data_folder + TRAINING_ATTRS_FILE2)


def generate_attr_triples_data(all_attr_triples_file, ents, out_file):
    lines = read_lines(all_attr_triples_file)
    file = open(out_file, 'w', encoding='utf8')
    ents_set = set()
    num = 0
    for line in lines:
        triple = parse_ttl_lines(line)
        # remove dbo triples
        if triple[0] in ents and DBO_PREFIX not in triple[1]:
            file.write(line)
            ents_set.add(triple[0])
            num += 1
    print("all attr triples", num)
    if len(ents_set) != len(ents):
        print(len(ents_set), len(ents))
        print("some ents have no attrs")
    file.close()


def filter_dbo_attrs(attrs):
    infobox_attrs = set()
    for attr in attrs:
        if DBO_PREFIX not in attr:
            infobox_attrs.add(attr)
    return infobox_attrs


def get_attrs(all_attrs_dict, uris_set):
    attrs_dict = dict()
    attrs_set = set()
    for uri in uris_set:
        if uri in all_attrs_dict:
            # very important!!!
            attrs = filter_dbo_attrs(all_attrs_dict.get(uri))
            attrs_dict[uri] = attrs
            attrs_set |= attrs
    print_line("total ents in training data:", len(uris_set), "and", len(attrs_dict), " ents has", len(attrs_set),
               " attrs")
    return attrs_dict, attrs_set


def read_attrs_range(file_path):
    dic = dict()
    lines = read_lines(file_path)
    for line in lines:
        line = line.strip()
        params = line.split('\t')
        assert len(params) == 2
        dic[params[0]] = int(params[1])
    return dic


if __name__ == '__main__':
    dic = {'d': 111, 'c': 12}
    print(sorted(dic.items(), key=lambda x: x[1], reverse=True)[0][0])

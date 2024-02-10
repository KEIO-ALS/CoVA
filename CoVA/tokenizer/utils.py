import Levenshtein
import os

def get_file_names(directory):
    file_names = []
    for file_name in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, file_name)):
            if str(file_name).endswith("vcf.gz"):
                file_names.append(file_name)
    return file_names

def get_seq_id(seq):
    base_id_list = {
        "n": 0,
        "N": 0,
        "a": 1,
        "A": 1,
        "t": 2,
        "T": 2,
        "g": 3,
        "G": 3,
        "c": 4,
        "C": 4,
    }
    return sum([4**(len(seq)-i-1)*base_id_list[base] for i, base in enumerate(seq)])

def get_diff(start_pos, string1, string2):
    differ = Levenshtein.editops(string1, string2)
    correction = 0
    for op, i, j in differ:
        index = int(start_pos)+i+correction
        diff = ""
        if op == 'replace':
            yield index, string1[i]+string2[j]
        elif op == 'delete':
            yield index, string1[i]+"N"
        elif op == 'insert':
            yield index, "N"+string2[j]
            correction +=1

def merge_dicts(dict_a, dict_b):
    merged_dict = {}
    for key in set(dict_a.keys()).union(set(dict_b.keys())):
        value_a = dict_a.get(key)
        value_b = dict_b.get(key)
        if value_a and value_b:
            merged_dict[key] = value_a + value_b
        elif value_a:
            merged_dict[key] = value_a + "NN"
        elif value_b:
            merged_dict[key] = "NN" + value_b
    return merged_dict

def get_info_from_line(line):
    chrA = {}
    chrB = {}
    data = line.split("\t")
    if len(data) == 10:
        chr_id, pos, _, ref, alts, qual, _, _, _, ss = data
    else:
        chr_id, pos, _, ref, alts, *_ = data
        ss = "1/0:"

    chr_id = int(chr_id.split(".")[0][-2:])
    for i, b in enumerate(ss.split(":")[0].split("/")):
        if bool(int(b)):
            alt = alts.split(",")[int(b)-1]
        else:
            alt = ref
        for index, diff in get_diff(pos, ref, alt):
            if i==0:
                chrA[index]=diff
            if i==1:
                chrB[index]=diff
    for diff in merge_dicts(chrA, chrB).items():
        # yield [chr_id, diff[0], diff[1], get_seq_id(diff[1]), float(qual)]
        yield [chr_id, diff[0], diff[1], get_seq_id(diff[1])]
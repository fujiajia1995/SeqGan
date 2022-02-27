# -*- utf-8 -*-

from collections import defaultdict
import numpy as np

file_address = "./all_data"


def word_count(file_address):
    result = defaultdict(int)
    ignore_list = []
    with open(file_address, "r") as f:
        for sentence in f:
            word_list = sentence.split(" ")
            for word in word_list:
                result[word] += 1
    for word, count in result.items():
        if count <= 21000:
            ignore_list.append(word)
    return result, ignore_list


def make_data(ignore, file_address):
    ignore_set = set(ignore)
    count = 0
    with open(file_address, "r") as r:
        with open("true_data", "w") as w:
            for sentence in r:
                word_set = set(sentence.split(" "))
                if len(word_set.intersection(ignore_set)) == 0 and len(word_set) > 10:
                    print(sentence, end="", file=w)
                    count += 1
    return count


if __name__ == "__main__":
    result, ignore_list = word_count(file_address)
    print("word_count:", end="")
    print(len(result.keys())-len(ignore_list))
    count = make_data(ignore_list, file_address)
    print("sentence_count:", end="")
    print(count)


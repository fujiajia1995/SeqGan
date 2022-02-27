# -*- utf-8 -*-

import spacy
import os
from tqdm import tqdm
import re
import string


raw_data_directory = "./raw_data"
raw_data_num = 8


tokenizer = spacy.load("en_core_web_lg")


def raw_process(name):
    count = 0
    try:
        with open(name, "r") as f:
            with open("all_data", "a") as wf:
                for essay in f:
                    sentence_list = re.split("[.]", essay)
                    for sentence in sentence_list:
                        token_list = tokenizer(sentence.lower())
                        if len(token_list) > 10:
                            count += 1
                            for token in token_list:
                                if token.text != "\n" and token.text != " " and token.text not in string.punctuation:
                                    print(token.text, end=" ", file=wf)
                            print(end="\n", file=wf)
    except:
        pass
    return count


if __name__ == "__main__":
    number = 0
    for _, _, file_list in os.walk(raw_data_directory):
        for index in tqdm(range(len(file_list))):
            i = file_list[index]
            tqdm.write(i)
            file_name = os.path.join(raw_data_directory, str(i))
            number += raw_process(file_name)
            tqdm.write(str(number))
    print(number)
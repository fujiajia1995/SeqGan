# -*- utf-8 -*-

import pickle5 as pickle
from vocabulary import Vocabulary
import torch
from config import config
import numpy as np


class Vectorizer(object):
    def __init__(self, vocabulary, max_len):
        self._vocabulary = vocabulary
        self.max_len = max_len

    def get_vocabulary(self):
        return self._vocabulary

    @classmethod
    def from_dataframe(cls, text_dataframe):
        vocabulary = Vocabulary()
        max_len = 0
        for _, row in text_dataframe.iterrows():
            token_list = row.data.split(" ")
            now_len = len(token_list)
            if now_len > max_len:
                max_len = now_len
            for token in token_list:
                vocabulary.add_token(token)
        return cls(vocabulary, max_len+2)

    def vectorize(self, sentence):
        result = np.zeros(self.max_len, dtype=np.int64)
        #indices = [self._vocabulary._begin_idx]
        indices = []
        for word in sentence.split(" ")[:-1]:
            word = word.lower()
            if word in self._vocabulary._token_to_idx.keys():
                indices.append(self._vocabulary.look_up_token(word))
            else:
                indices.append(self._vocabulary._unknown_idx)
        #indices.append(self._vocabulary._end_idx)
        result[:len(indices)] = indices
        result = torch.tensor(result).to(config.device)
        return result

    def reverse_list(self,input):
        result = []
        for i in range(len(input)):
            result.append(self._vocabulary.look_up_index(input[i]))
        return " ".join(result)


if __name__ == "__main__":
    with open("./test_file/test.pickle", "rb") as f:
        text_data = pickle.load(f)
    test_vectorizer = Vectorizer.from_dataframe(text_data)
    print(len(test_vectorizer._vocabulary))
    print(test_vectorizer.vectorize("i have a dream that is happy"))
    print(test_vectorizer._vocabulary._idx_to_token[2])
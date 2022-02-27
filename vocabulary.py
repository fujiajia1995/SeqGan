# -*- utf-8-*-

from config import config
import pickle5 as pickle


class Vocabulary(object):
    def __init__(self, token_to_idx=None):
        if token_to_idx is None:
            token_to_idx = {}
        self._token_to_idx = token_to_idx
        self._idx_to_token = {idx: token for token, idx in self._token_to_idx.items()}
        self._padding_token = "<padding>"
        self._padding_idx = self.add_token(self._padding_token)
        self._unk_token = config.unknown_token
        self._unknown_idx = self.add_token(config.unknown_token)
        self._begin_token = config.begin_token
        self._begin_idx = self.add_token(config.begin_token)
        self._end_token = config.end_token
        self._end_idx = self.add_token(config.end_token)

    def add_token(self, token):
        token = token.lower()
        if token in self._token_to_idx:
            return self._token_to_idx[token]
        else:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token
        return index

    def look_up_token(self, token):
        return self._token_to_idx.get(token, self._unknown_idx)

    def look_up_index(self, index):
        if index not in self._idx_to_token.keys():
            raise KeyError("no " + str(index) + " in dict")
        return self._idx_to_token[index]

    def to_pickle(self, name):
        with open(name, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_from_pickle(name):
        with open(name, "rb") as f:
            return pickle.load(f)

    def __str__(self):
        return "vocabulary size: " + str(len(self._token_to_idx))

    def __len__(self):
        return len(self._token_to_idx.keys())


# -*- utf-8 -*-

from generator import Generator
from classficater import Discriminator
import numpy as np
import copy
import torch
from config import config
from tqdm import tqdm


class Node:
    def __init__(self, sequence=[0]):
        assert type(sequence) == type([])
        self.stat = sequence
        self.value = 0


class MonteCarloSearchTree:
    def __init__(self, node, generate, discriminator, seq_len,g_vectorize,d_vectorize):
        self.root = node
        self.current = copy.deepcopy(self.root.stat)
        self.seq_len = seq_len
        self.generate = generate
        self.vocabulary = self.generate.vectorize_size
        self.discriminator = discriminator
        self.g_vectorize =g_vectorize
        self.d_vectorize = d_vectorize

    def take_action(self, action):
        self.current.append(action)

    def clear_current(self):
        self.current.pop(-1)

    def rollout(self, generate_len):
        assert self.current is not None
        if generate_len == 0:
            result = self.current
        else:
            result = self.generate.random_generate(input_node=self.current, num=generate_len)
        return result

    def get_value(self, n):
        result = []
        for i in range(self.vocabulary):
            self.take_action(i)
            score = 0
            for j in range(n):
                sentence = self.rollout(self.seq_len-1)
                sentence = self.g_vectorize.reverse_list(sentence)
                sentence = self.d_vectorize.vectorize(sentence)
                sentence = sentence.unsqueeze(0).to(config.device)
                #print(self.discriminator(sentence)[0][1].item())
                score += self.discriminator(sentence)[0][1].item()
            score = score/n
            self.clear_current()
            result.append(score)
        return torch.tensor(result, requires_grad=False).to(config.device)


if __name__ == "__main__":
    G = Generator(embedding_size=50, num_layers=3, hidden_size=50, vectorize_size=100, embedding_weight=None)
    D = Discriminator(embedding_size=50, vectorizer_size=100)
    root = Node([1, 2])
    tree = MonteCarloSearchTree(node=root, generate=G, discriminator=D, seq_len=10)
    score = tree.get_value(6)
    print(score.size())
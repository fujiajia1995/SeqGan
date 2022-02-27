# -*- utf-8 -*-

from generator import Generator
from config import config
import torch


model_stat = torch.load("./model/GAN.ckpt",map_location=config.device)
dataset = model_stat["gen_dataset"]
vectorize = dataset.get_vectorizer()
vocabulary = vectorize.get_vocabulary()

model = Generator(embedding_size=config.embedding_size,
                  num_layers=config.num_layers,
                  hidden_size=config.init_gen_hidden_size,
                  vectorize_size=len(vocabulary),
                  embedding_weight=None)
model.load_state_dict(model_stat["gen_model_stat"])

sentence = "he pointed out "
sentence = sentence.split(" ")

sentence_list = []
for i in range(len(sentence)):
    sentence_list.append(vocabulary.look_up_token(sentence[i]))


print(vectorize.reverse_list(model.random_generate(sentence_list, num=18)))

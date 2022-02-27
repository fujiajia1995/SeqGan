# -*- utf-8 -*-

import argparse

config = argparse.Namespace()

config.true_data_address = "./data/true_data"
config.false_data_address = "./data/false_data"
config.init_model = "./model/init_generator"
config.embedding_file = "./model/glove.6B.200d.txt"

config.train_data = 0.9
config.dev_data = 0.1
config.padding_index = 0

config.device = "cpu"

config.init_epoch = 20000
config.init_batch_size = 700
config.init_lr_rate = 0.003
config.init_gen_hidden_size = 1500
config.embedding_size = 200
config.num_layers = 3

config.weight_decay = 0.003
config.dropout_p = 0.2
config.clip = 0.001

config.classification_epoch = 200
config.classification_data_num = 15000
config.classification_batch_size = 500
config.classification_lr = 0.00008
config.classification_clip = 0.1
config.cnn_out_channel = 200
config.cnn_kernel_len = 3

config.gan_epoch = 10000
config.G_step = 2
config.D_step = 4
config.Gan_epoch =1000
config.Gan_gen_lr = 0.0003
config.Gan_Dis_lr = 0.03
config.random_count = 6


config.unknown_token = "<unk>"
config.begin_token = "<bos>"
config.end_token = "<eos>"





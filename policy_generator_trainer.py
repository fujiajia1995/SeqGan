# -*- utf-8 -*-

from classfication_trainer import ClassificationTrainer
from init_generator_trainer import InitGenTrainer
from config import config
import torch
from tqdm import tqdm
import numpy as np
import os
from MCST import MonteCarloSearchTree
from MCST import Node

save_epoch = 2
save_path_dir = "./checkpoint/GAN_G"
file_name_root = "polict_generator"


def make_init_generator_trainer(epoch, batch_size,
                                embedding_size,
                                num_layers,
                                hidden_size,
                                lr,
                                train=False):
    return InitGenTrainer(epoch=epoch,
                          batch_size=batch_size,
                          embedding_size=embedding_size,
                          num_layers=num_layers,
                          hidden_size=hidden_size,
                          lr=lr,
                          train=train)


def make_init_classification_trainer(epoch, batch_size, lr, embedding_size, clip, train=False):
    return ClassificationTrainer(epoch=epoch,
                                 batch_size=batch_size,
                                 lr=lr,
                                 embedding_size=embedding_size,
                                 clip=clip,
                                 train=train)


class GanTrainer(object):
    def __init__(self, epoch, G_step, D_step, generator_ckpt_address=None, classification_ckpt_address=None):
        self.epoch = epoch
        self.G_step = G_step
        self.D_step = D_step
        self.classification_trainer = make_init_classification_trainer(epoch=config.classification_epoch,
                                                                       batch_size=config.classification_batch_size,
                                                                       lr=config.classification_lr,
                                                                       embedding_size=config.embedding_size,
                                                                       clip=config.classification_clip,
                                                                       train=False)
        self.generator_trainer = make_init_generator_trainer(epoch=config.classification_epoch,
                                                             batch_size=config.init_batch_size,
                                                             embedding_size=config.embedding_size,
                                                             num_layers=config.num_layers,
                                                             hidden_size=config.init_gen_hidden_size,
                                                             lr=config.init_lr_rate,
                                                             train=False)
        if generator_ckpt_address:
            self.generator_trainer.load_model_from_path(generator_ckpt_address)
        else:
            self.generator_trainer.train()
            print("generator init over, please input the ckpt address and re try")
            exit()
        if classification_ckpt_address:
            self.classification_trainer.load_model(classification_ckpt_address)
        self.discriminator = self.classification_trainer.model
        self.gen_optimizer = torch.optim.Adam(self.generator_trainer.model.parameters(), lr=config.Gan_gen_lr)

    def _train(self, start_num=0):
        self.classification_trainer.model.to(config.device)
        self.generator_trainer.model.to(config.device)
        for epoch in tqdm(range(start_num, self.epoch)):  # epoch
            tqdm.write("gan_epoch start")
            self.generator_trainer.model.train()
            for g_step in range(self.G_step):  # g_step
                self.gen_optimizer.zero_grad()
                # num = np.random.choice(range(10, 14), 1)[0]
                num = 3
                sentence = self.generator_trainer.model.random_generate(num=num)
                sentence_len = len(sentence)
                sentence_tensor = torch.tensor(sentence).unsqueeze(0).to(config.device)
                score_list = []
                # print(num)
                for index in range(1, sentence_len):
                    node = Node(sentence[:index])
                    score = MonteCarloSearchTree(node=node, generate=self.generator_trainer.model,
                                                 discriminator=self.classification_trainer.model,
                                                 seq_len=(sentence_len - index),
                                                 g_vectorize=self.generator_trainer._vectorize,
                                                 d_vectorize=self.classification_trainer._vectorize).get_value(6)
                    score_list.append(score)
                score = torch.stack(score_list).to(config.device)
                generator_probability = self.generator_trainer.model(sentence_tensor).squeeze(0)[:-1].to(config.device)
                loss = -torch.sum(torch.mean(score * generator_probability, dim=-1).to(config.device))
                loss.backward()
                self.gen_optimizer.step()
            self.generator_trainer.model.eval()
            with open("./log.temp", "w")as f:
                print(epoch, file=f)
                print(self.generator_trainer._vectorize.reverse_list(
                    self.generator_trainer.model.random_generate(num=20)), file=f)
            print(self.generator_trainer._vectorize.reverse_list(self.generator_trainer.model.random_generate(num=20)))

            self.classification_trainer.model.train()
            for D_step in range(self.D_step):
                self.generator_trainer.make_false_data(num=15000)
                self.classification_trainer.train()
            self.classification_trainer.model.eval()

            if epoch % save_epoch == 0:
                file_name = file_name_root + str(epoch)
                path = os.path.join(save_path_dir, file_name)
                self.save_model(epoch=epoch, path=path)

    def train(self):
        file_list = os.listdir(save_path_dir)
        file_list_len = len(file_list)
        if file_list_len != 0:
            load_file = file_name_root + str((file_list_len - 1) * save_epoch)
            load_file_path = os.path.join(save_path_dir, load_file)
            epoch = self.load_model_from_path(load_file_path)
            self._train(start_num=epoch + 1)
        else:
            self._train()

    def save_model(self, epoch, path):
        torch.save({
            "dataset_operator": self.dataset,
            "gen_model_stat": self.generator_trainer.model.state_dict(),
            "optimizer": self.gen_optimizer.state_dict(),
            "classifier_model_stat": self.classification_trainer.model.state_dict(),
            "epoch": epoch
        }, path)

    def load_model_from_path(self, path):
        model_stat = torch.load(path, map_location=config.device)
        self.classification_trainer.model.load_state_dict(model_stat["classifier_model_stat"])
        self.generator_trainer.model.load_state_dict(model_stat["gen_model_stat"])
        self.gen_optimizer.load_state_dict(model_stat["optimizer"])
        return model_stat["epoch"]


if __name__ == "__main__":
    policy_trainer = GanTrainer(generator_ckpt_address="./model/init/init_generator.ckpt",
                                classification_ckpt_address="./model/init/init_classification.ckpt",
                                epoch=1,
                                G_step=1,
                                D_step=1)
    policy_trainer._train()
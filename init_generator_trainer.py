# -*- utf-8 -*-

from generator import Generator
from config import config
from dataset import InitDataSet
from load_pretrain_embedding import PreTrainEmbedding
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import os
import numpy as np

predict_load_file_address = "./checkpoint/init_G/init_checkpoint"
checkpoint_name_root = "init_checkpoint"
checkpoint_dir_name = "./checkpoint/init_G"


class InitGenTrainer(object):
    def __init__(self, epoch, batch_size, embedding_size, num_layers, hidden_size, lr, train=True):
        self.epoch = epoch
        self.lr = lr
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.dataset = InitDataSet.load_from_text(config.true_data_address)
        self._vectorize = self.dataset.get_vectorizer()
        self._vocabulary = self._vectorize.get_vocabulary()
        self.vectorize_size = len(self._vocabulary)
        if train:
            embedding_weight = PreTrainEmbedding.load_from_file(self._vocabulary, config.embedding_file)
        else:
            embedding_weight = None
        self.model = Generator(embedding_size=embedding_size,
                               num_layers=num_layers,
                               hidden_size=hidden_size,
                               vectorize_size=self.vectorize_size,
                               embedding_weight=embedding_weight).to(config.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def _generate_data(self):
        data_loader = DataLoader(dataset=self.dataset, batch_size=self.batch_size,
                                 shuffle=True, drop_last=True)

        for data_dict in data_loader:
            out_data = {}
            for name, tensor in data_dict.items():
                out_data[name] = data_dict[name].to(config.device)
            yield out_data

    def save_model(self, epoch, loss, path):
        torch.save({
            "dataset_operator": self.dataset,
            "model_stat": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "loss": loss,
            "epoch": epoch
        }, path)

    def load_model_from_path(self, path):
        model_stat = torch.load(path, map_location=config.device)
        self.model.load_state_dict(model_stat["model_stat"])
        self.optimizer.load_state_dict(model_stat["optimizer"])
        self.dataset = model_stat["dataset_operator"]
        loss = model_stat["loss"]
        epoch = model_stat["epoch"]
        return epoch, loss

    def _train(self, start_point=0):
        for epoch_index in tqdm(range(start_point, self.epoch)):
            batch_generator = self._generate_data()
            self.model.train()
            epoch_loss = 0
            for batch_index, batch_data in enumerate(batch_generator):
                self.optimizer.zero_grad()
                input_data = batch_data["x_sequence"]
                y_pred = self.model(input_data).to(config.device)
                feature_size = y_pred.size(-1)
                y_pred = y_pred[:, :-1, :].contiguous().view(-1, feature_size).to(config.device)
                x_target = input_data[:, 1:].contiguous().view(-1).to(config.device)
                loss = self.loss(y_pred, x_target)
                epoch_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.clip)
                self.optimizer.step()
            self.model.eval()
            checkpoint_name = checkpoint_name_root + str(epoch_index)
            path = os.path.join(checkpoint_dir_name, checkpoint_name)
            self.save_model(epoch_index, loss, path)
            print(epoch_loss/(batch_index+1))
            print(self._vectorize.reverse_list(self.model.random_generate(num=20)))

    def train(self):
        file_list = os.listdir(checkpoint_dir_name)
        file_list_len = len(file_list)
        if file_list_len != 0:
            load_file = checkpoint_name_root+str(file_list_len-1)
            load_file_path = os.path.join(checkpoint_dir_name, load_file)
            epoch, _ = self.load_model_from_path(load_file_path)
            self._train(start_point=epoch+1)
        else:
            self._train()

    def make_false_data(self, address=None, num=1):
        if address:
            self.load_model_from_path(address)
        else:
            pass
        self.model.eval()
        with open("./data/false_data", "w") as f:
            for i in tqdm(range(num)):
                num = np.random.choice(range(10, 20), 1)[0]
                print(self._vectorize.reverse_list(self.model.random_generate(num=num)), file=f)

    def get_model(self):
        return self.model


if __name__ == "__main__":
    generator_trainer_1 = InitGenTrainer(epoch=config.classification_epoch,
                                         batch_size=config.init_batch_size,
                                         embedding_size=config.embedding_size,
                                         num_layers=config.num_layers,
                                         hidden_size=config.init_gen_hidden_size,
                                         lr=config.init_lr_rate,
                                         train=False)

    generator_trainer_1.make_false_data(address="./model/init/init_checkpoint", num=15000)

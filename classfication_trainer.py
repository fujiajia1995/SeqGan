# -*- utf-8 -*-

from classficater import Discriminator
from config import config
from dataset import ClassificationDataSet
from load_pretrain_embedding import PreTrainEmbedding
from torch.utils.data import DataLoader
import torch
import torch.nn
from tqdm import tqdm
import os

predict_load_file_address = "./checkpoint/GAN_D/D_classification.ckpt"
checkpoint_name = "D_classification.ckpt"
checkpoint_dir_name = "./checkpoint/GAN_D"


class ClassificationTrainer(object):
    def __init__(self, epoch, batch_size,  lr, embedding_size,  clip, train=True):
        self.loss = torch.nn.BCELoss()
        self.clip = clip
        self.dataset = ClassificationDataSet.load_from_text(true_data_address=config.true_data_address,
                                                            false_data_address=config.false_data_address,
                                                            num=config.classification_data_num)
        self.word_num = None
        self._vectorize = self.dataset.get_vectorizor()
        self._vocabulary = self._vectorize.get_vocabulary()
        self.vectorize_len = len(self._vocabulary)
        if train:
            embedding_weight = PreTrainEmbedding.load_from_file(self._vocabulary, config.embedding_file)
        else:
            embedding_weight = None
        self.model = Discriminator(embedding_size=embedding_size,
                                   vectorizer_size=self.vectorize_len,
                                   embedding_weight=embedding_weight)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.batch_size = batch_size
        self.epoch = epoch

    def _data_generator(self):
        batch_data_loader = DataLoader(dataset=self.dataset, batch_size=self.batch_size,
                                       shuffle=True, drop_last=True)
        return batch_data_loader

    def _train(self, start_num=0):
        self.model.to(config.device)
        precesion = 0
        for epoch_index in tqdm(range(start_num, config.classification_epoch)):
            self.dataset.set_split("train")
            batch_data_loader = self._data_generator()
            self.model.train()
            total_loss = 0
            for batch_index, data in enumerate(batch_data_loader):
                x_input = data["x_data"].to(config.device)
                y_target = data["y_target"].to(config.device).to(torch.float)
                self.optimizer.zero_grad()
                y_pred = self.model(x_input).to(config.device)
                loss = self.loss(y_pred[:, 1], y_target)
                loss.backward()
                total_loss += loss.item()
                self.optimizer.step()
            tqdm.write(str(total_loss))

            self.model.eval()
            self.dataset.set_split("dev")
            test_batch_data_loader = self._data_generator()
            batch_correct_count = 0
            for batch_index, data in enumerate(test_batch_data_loader):
                x_input = data["x_data"].to(config.device)
                y_target = data["y_target"].to(config.device).to(torch.float)
                y_pred = self.model(x_input).to(config.device)

                for i in range(y_pred.size(0)):
                    if y_pred[i][1].item() > y_pred[i][0].item():
                        result = 1
                    else:
                        result = 0
                    if int(result) == int(y_target[i].item()):
                        batch_correct_count += 1
            current_precesion = batch_correct_count/len(self.dataset)
            tqdm.write(str(current_precesion))
            if current_precesion > precesion:
                precesion = current_precesion
                path = os.path.join(checkpoint_dir_name, checkpoint_name)
                self.save_model(epoch_index, loss=total_loss, path=path)

    def save_model(self, epoch, loss, path):
        torch.save({
            "model_stat": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "loss": loss,
            "epoch": epoch,
            "dataset": self.dataset
        }, path)

    def load_model(self, path):
        model_stat = torch.load(path)
        self.model.load_state_dict(model_stat["model_stat"])
        self.optimizer.load_state_dict(model_stat["optimizer"])
        self.dataset = model_stat["dataset"]
        return model_stat["loss"], model_stat["epoch"]

    def train(self):
        file_list = os.listdir(checkpoint_dir_name)
        file_list_len = len(file_list)
        if file_list_len != 0:
            load_file_path = os.path.join(checkpoint_dir_name, checkpoint_name)
            epoch, _ = self.load_model(load_file_path)
            self._train(start_num=int(epoch)+1)
        else:
            self._train()

    def get_model(self):
        return self.model

    def reload_dataset(self):
        self.dataset = ClassificationDataSet.load_from_text(true_data_address=config.true_data_address,
                                                            false_data_address=config.false_data_address,
                                                            num=config.classification_data_num)


if __name__ == "__main__":
    classification_trainer = ClassificationTrainer(epoch=config.classification_epoch,

                                                   batch_size=config.classification_batch_size,
                                                   lr=config.classification_lr,
                                                   embedding_size=config.embedding_size,
                                                   clip=config.classification_clip)
    classification_trainer.train()
# -*- utf-8 -*-
from config import config
from utils.make_dataframe_from_txt import DataframeMaker
from vectorizer import Vectorizer
from config import config
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch


class ClassificationDataSet(Dataset):
    def __init__(self, dataframe, vectorizer):
        self._target_data_df = None
        self._target_len = None
        self.split = None

        self.all_data_df = dataframe
        self.vectorizer = vectorizer
        self.len = len(self.all_data_df)

        random_list = np.random.permutation(self.len)
        temp = int(self.len * config.train_data)
        train_index = random_list[:temp]
        dev_index = random_list[temp:]

        self.train_data_df = self.all_data_df.iloc[train_index]
        self.train_len = len(self.train_data_df)

        self.dev_data_df = self.all_data_df.iloc[dev_index]
        self.dev_len = len(self.dev_data_df)

        self.look_up_dict = {
            "all": (self.all_data_df, self.len),
            "train": (self.train_data_df, self.train_len),
            "dev": (self.dev_data_df, self.dev_len)
        }

        self.set_split("all")

    def get_vectorizor(self):
        return self.vectorizer

    @classmethod
    def load_from_text(cls, true_data_address, false_data_address, num):
        maker = DataframeMaker()
        maker.load_data(true_data_address, "true")
        maker.load_data(false_data_address, "false")
        maker.true_dataframe = maker.true_dataframe.sample(n=num, random_state=1, replace=False)
        maker.false_dataframe = maker.false_dataframe.sample(n=num, random_state=1, replace=False)
        dataframe = maker.make_dataframe()
        vectorizer = Vectorizer.from_dataframe(maker.true_dataframe)
        return cls(dataframe, vectorizer)

    def __getitem__(self, index):
        row = self._target_data_df.iloc[index]
        indices = self.vectorizer.vectorize(row.data).to(config.device)
        return {
            "x_data": indices,
            "y_target": row.label
        }

    def set_split(self, split="all"):
        self.split = split
        self._target_data_df, self._target_len = self.look_up_dict[split]

    def __len__(self):
        return len(self._target_data_df)


class InitDataSet(Dataset):
    def __init__(self, dataframe, vectorizer):
        self.data_df = dataframe
        self.vectorizer = vectorizer

    @classmethod
    def load_from_text(cls, true_data_address):
        maker = DataframeMaker()
        maker.load_data(true_data_address, "true")
        dataframe = maker.make_dataframe()
        vectorizer = Vectorizer.from_dataframe(maker.true_dataframe)
        return cls(dataframe, vectorizer)

    def get_vectorizer(self):
        return self.vectorizer

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        row = self.data_df.iloc[index]
        indices = self.vectorizer.vectorize(row.data).to(config.device)
        return {"x_sequence": indices}


if __name__ == "__main__":
    test_class_dataset = ClassificationDataSet.load_from_text(config.true_data_address, config.false_data_address)
    test_class_dataset.set_split("all")
    test_init_dataset = InitDataSet.load_from_text(config.true_data_address)
    dataloader = DataLoader(dataset=test_init_dataset, batch_size=50,
                            shuffle=True, drop_last=True)

    for data in dataloader:
        print(data["x_sequence"])
        break

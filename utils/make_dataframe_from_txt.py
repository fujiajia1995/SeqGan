# -*- utf-8 -*-

import pandas as pd


class DataframeMaker(object):
    def __init__(self):
        self.true_dataframe = None
        self.false_dataframe = None
        self.have_true_data = False
        self.have_false_data = False
        self.true_data_num = 0
        self.false_data_num = 0

    def load_data(self, address, mode):
        assert mode in ["true", "false"]
        sentence_list = []
        with open(address, "r") as f:
            for sentence in f:
                sentence_list.append(sentence)
        data_len = len(sentence_list)
        if mode == "true":
            label_list = [1] * data_len
            self.true_dataframe = pd.DataFrame(
                data={"data": sentence_list,
                      "label": label_list},
                columns=["data", "label"]
            )
            self.have_true_data = True
            self.true_data_num = data_len
        else:
            label_list = [0] * data_len
            self.false_dataframe = pd.DataFrame(
                data={"data": sentence_list,
                      "label": label_list},
                columns=["data", "label"]
            )
            self.have_false_data = True
            self.false_data_num = data_len

    def make_dataframe(self):
        output = self.true_dataframe.append(self.false_dataframe)
        output = output.sample(frac=1).reset_index(drop=True)
        return output


if __name__ == "__main__":
    true_data_address = "../data/true_data_7"
    false_data_address = "../data/false_data"
    test = DataframeMaker()
    test.load_data(true_data_address, "true")
    #test.load_data(false_data_address, "false")
    result = test.make_dataframe()
    print(len(result))



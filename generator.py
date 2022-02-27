from torch import nn
import torch
from torch.nn import LSTMCell
from config import config
import numpy as np


class MultilayerLstmCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_layer):
        super(MultilayerLstmCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.lstm_unit_list = []
        self.lstm_unit_list.append(LSTMCell(input_size=input_size,
                                            hidden_size=hidden_size).to(config.device))
        for i in range(num_layer-1):
            self.lstm_unit_list.append(LSTMCell(input_size=hidden_size,
                                                hidden_size=hidden_size).to(config.device))
        self.lstm_unit_list = nn.ModuleList(self.lstm_unit_list)

    def forward(self, x_input, y_hidden_list, cell_stat_list):
        result_hidden_stat = []
        result_cell_stat = []
        for i in range(self.num_layer):
            hidden_stat, cell_stat = self.lstm_unit_list[i](x_input,
                                                            (y_hidden_list[i], cell_stat_list[i]))
            hidden_stat = hidden_stat.to(config.device)
            cell_stat = cell_stat.to(config.device)
            x_input = hidden_stat
            result_cell_stat.append(cell_stat)
            result_hidden_stat.append(hidden_stat)
        return result_hidden_stat, result_cell_stat


class Generator(nn.Module):
    def __init__(self, embedding_size, num_layers, hidden_size,
                 vectorize_size, embedding_weight=None):
        super(Generator, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.vectorize_size = vectorize_size
        self.embedding_layer = nn.Embedding(num_embeddings=vectorize_size,
                                            embedding_dim=embedding_size,
                                            _weight=embedding_weight)
        self.lstm_layer = MultilayerLstmCell(input_size=embedding_size,
                                             hidden_size=hidden_size,
                                             num_layer=num_layers).to(config.device)
        self.output_linear = torch.nn.Linear(in_features=self.hidden_size, out_features=self.vectorize_size)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(p=config.dropout_p)

    def _init_hidden_stat(self, batch_size):
        hidden_stat = torch.nn.init.xavier_normal_(torch.zeros(batch_size, self.hidden_size))
        return hidden_stat.to(config.device)

    def _init_cell_stat(self, batch_size):
        cell_stat = torch.nn.init.xavier_normal_(torch.zeros(batch_size, self.hidden_size))
        return cell_stat.to(config.device)

    def forward(self, y_target):
        batch_size, sequence_len = y_target.size()
        result = []
        hidden_stat_list = []
        for i in range(self.num_layers):
            hidden_stat_list.append(self._init_hidden_stat(batch_size))
        cell_stat_list = []
        for i in range(self.num_layers):
            cell_stat_list.append(self._init_cell_stat(batch_size))
        y_target = y_target.permute(1, 0)
        for i in range(sequence_len):
            lstm_input = self.embedding_layer(y_target[i]).to(config.device)
            hidden_stat_list, cell_stat_list = self.lstm_layer(lstm_input, hidden_stat_list, cell_stat_list)
            probability_hidden_stat = self.dropout(hidden_stat_list[-1]).to(config.device)
            probability = self.softmax(self.output_linear(probability_hidden_stat))
            result.append(probability)
        result = torch.stack(result, dim=0).permute(1, 0, 2)
        return result

    def random_generate(self, input_node=None, num=1):
        result = []
        cell_stat_list = []
        for i in range(self.num_layers):
            cell_stat_list.append(self._init_cell_stat(1))
        hidden_stat_list = []
        for i in range(self.num_layers):
            hidden_stat_list.append(self._init_hidden_stat(1))
        if input_node is None:
            input_node = np.random.choice(range(self.vectorize_size), 1)
        result.extend(input_node)
        input_node = torch.tensor(input_node).unsqueeze(0)
        input_node = input_node.permute(1, 0).to(config.device)
        length = input_node.size(0)
        for i in range(length):
            lstm_input = self.embedding_layer(input_node[i]).to(config.device)
            hidden_stat_list, cell_stat_list = self.lstm_layer(lstm_input, hidden_stat_list, cell_stat_list)
        for i in range(num):
            generate_input_probability = self.softmax(self.output_linear(hidden_stat_list[-1]))
            probability = generate_input_probability.squeeze().to("cpu").detach().numpy().copy()
            generate_node = np.random.choice(range(self.vectorize_size), 1, p=probability, replace=False)
            #print(generate_node)
            #random_choice = np.random.choice(range(5), 1)
            #print(random_choice)
            #generate_node = [generate_node[random_choice[0]]
            result.append(generate_node[0])
            generate_input = torch.tensor(generate_node).to(config.device)
            generate_input = self.embedding_layer(generate_input).to(config.device)
            hidden_stat_list, cell_stat_list = self.lstm_layer(generate_input, hidden_stat_list, cell_stat_list)
        return result

    def beam_generate(self, input_node=None, num=1):
        result = []
        cell_stat_list = []
        for i in range(self.num_layers):
            cell_stat_list.append(self._init_cell_stat(1))
        hidden_stat_list = []
        for i in range(self.num_layers):
            hidden_stat_list.append(self._init_hidden_stat(1))
        if input_node is None:
            input_node = np.random.choice(range(self.vectorize_size), 1)
        result.extend(input_node)
        input_node = torch.tensor(input_node).unsqueeze(0)
        input_node = input_node.permute(1, 0).to(config.device)
        length = input_node.size(0)
        for i in range(length):
            lstm_input = self.embedding_layer(input_node[i]).to(config.device)
            hidden_stat_list, cell_stat_list = self.lstm_layer(lstm_input, hidden_stat_list, cell_stat_list)
        for i in range(num):
            generate_input_probability = self.softmax(self.output_linear(hidden_stat_list[-1]))
            probability = generate_input_probability.squeeze().to("cpu").detach().numpy().copy()
            choice = np.random.choice(range(3), 1)[0]
            generate_node = np.random.choice(range(self.vectorize_size), 3, p=probability, replace=False)
            #print(generate_node)
            #random_choice = np.random.choice(range(5), 1)
            #print(random_choice)
            #generate_node = [generate_node[random_choice[0]]
            result.append(generate_node[choice])
            generate_input = torch.tensor([generate_node[choice]]).to(config.device)
            generate_input = self.embedding_layer(generate_input).to(config.device)
            hidden_stat_list, cell_stat_list = self.lstm_layer(generate_input, hidden_stat_list, cell_stat_list)
        return result


if __name__ == "__main__":
    test_1 = Generator(embedding_size=50, num_layers=3, hidden_size=50, vectorize_size=100, embedding_weight=None)
    test_2 = Generator(embedding_size=50, num_layers=3, hidden_size=50, vectorize_size=100, embedding_weight=None)
    for p in test_2.state_dict():
        print(p)
    #for k, v in test_2.lstm_layer.named_parameters():
    #    print(k, v.size())

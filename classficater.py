# -*- utf-8 -*-

import torch.nn as nn
import torch
from config import config


class Discriminator(nn.Module):
    def __init__(self, embedding_size, vectorizer_size, embedding_weight=None):
        super(Discriminator, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vectorizer_size,
                                      embedding_dim=embedding_size,
                                      _weight=embedding_weight,
                                      padding_idx=config.padding_index)
        self.cnn_activate_func = torch.nn.ELU()
        self.cnn_unit = nn.Conv2d(in_channels=1, out_channels=config.cnn_out_channel,
                                  kernel_size=(config.cnn_kernel_len, embedding_size))
        self.pool_layer = torch.nn.AdaptiveMaxPool1d(1)

        self.w_t = nn.Parameter(torch.randn(config.cnn_out_channel, config.cnn_out_channel), requires_grad=True).\
            to(config.device)
        self.register_parameter("w_t", self.w_t)
        self.b_t = nn.Parameter(torch.randn(config.cnn_out_channel), requires_grad=True).to(config.device)
        self.register_parameter("b_t", self.b_t)
        self.w_h = nn.Parameter(torch.randn(config.cnn_out_channel, config.cnn_out_channel), requires_grad=True).\
            to(config.device)
        self.register_parameter("w_h", self.w_h)
        self.highway_activate_func = torch.nn.ELU()

        #self.output_w_o = nn.Parameter(torch.randn(config.cnn_out_channel, 100), requires_grad=True).to(config.device)
        #self.output_b_o = nn.Parameter(torch.randn(100), requires_grad=True).to(config.device)
        self.output_linear = torch.nn.Linear(in_features=config.cnn_out_channel, out_features=2)
        self.softmax = torch.nn.Softmax(dim=-1)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x_input):
        x_input = self.embedding(x_input).unsqueeze(1)
        y = self.cnn_activate_func(self.cnn_unit(x_input)).squeeze(-1)
        y = self.pool_layer(y).squeeze(-1).to(config.device)
        remember_gate = self.highway_activate_func(torch.matmul(y, self.w_t) + self.b_t).to(config.device)
        transformer_gate = self.highway_activate_func(torch.matmul(y, self.w_h)).to(config.device)
        efficient = 1-remember_gate
        efficient = efficient.to(config.device)
        y = torch.mul(remember_gate, transformer_gate) + torch.mul(efficient, y)
        y = self.softmax(self.sigmoid(self.output_linear(y)))
        return y


if __name__ == "__main__":
    D = Discriminator(embedding_size=30, vectorizer_size=40)
    y_init_target = torch.randint(low=0, high=30, size=(1, 3)).to(config.device)
    for k,v in D.named_parameters():
        print(k,v.size())


    #d_classification_input = torch.randint(low=1, high=20, size=(1, 5))
    #print(d_classification_input.size())
    #print(D(d_classification_input).size())


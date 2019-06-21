#-*-coding:utf-8-*-

# 特征编码器

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# 特征编码器
# TODO 关于年龄的文本切分存在问题，而且下一步也要把文本转化为数值
class FeaturesEncoder(nn.Module):

    def __init__(self, input_words_num, input_hidden_size, rnn_layers=1, rnn_bidirectional=True,
                 rnn_dropout=0):
        super(FeaturesEncoder, self).__init__()

        self.input_words_num = input_words_num
        self.input_hidden_size = input_hidden_size
        self.rnn_layers = rnn_layers
        self.rnn_dropout = rnn_dropout

        self.Embedding_layer = nn.Embedding(input_words_num, input_hidden_size)
        # TODO 载入预训练好的词向量

        self.RNN_layer = nn.GRU(input_size=input_hidden_size,
                                hidden_size=input_hidden_size,
                                num_layers=rnn_layers,
                                dropout=rnn_dropout,
                                bidirectional=rnn_bidirectional)
        # TODO 使用多层LSTM结构构件RNN计算层

    def forward(self, input_sequences, input_lengths, hidden=None):
        words_embedding = self.Embedding_layer(input_sequences)  # 从词向量层找出每一个单词的词向量

        packed = pack_padded_sequence(words_embedding, input_lengths)  # 根据每一个输入文本的长度压紧词向量

        outputs, hidden = self.RNN_layer(packed, hidden)  # 送入RNN中计算
        # TODO 如果RNN是LSTM结构，那么就不仅仅有hidden，还有cell_state，需要把hidden改为state

        outputs, _ = pad_packed_sequence(outputs)  # 解压还原

        outputs = outputs[:, :, :self.input_hidden_size] + outputs[:, :, self.input_hidden_size:] # 正向和反向叠加

        return outputs, hidden

# TODO 如果把病人的特征信息不是文本而是数值化，特征编码器就需要修改
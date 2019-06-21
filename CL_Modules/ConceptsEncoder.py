#-*-coding:utf-8-*-

# 概念编码器

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from Parameters import Parameters
param = Parameters()


# 概念编码器
class ConceptsEncoder(nn.Module):

    def __init__(self, input_words_num, input_hidden_size, rnn_layers=2, rnn_bidirectional=True,
                 rnn_dropout=0.5):
        super(ConceptsEncoder, self).__init__()

        self.input_words_num = input_words_num
        self.input_hidden_size = input_hidden_size
        self.rnn_layers = rnn_layers
        self.rnn_dropout = rnn_dropout

        self.Embedding_layer = nn.Embedding(input_words_num, input_hidden_size)
        # TODO 载入预训练好的词向量

        # self.RNN_layer = nn.GRU(input_size=input_hidden_size,
        #                         hidden_size=input_hidden_size,
        #                         num_layers=rnn_layers,  # LSTM使用两层结构
        #                         dropout=rnn_dropout,
        #                         bidirectional=rnn_bidirectional)

        self.RNN_layer = nn.LSTM(input_size=input_hidden_size,
                                 hidden_size=input_hidden_size,
                                 num_layers=rnn_layers,  # LSTM使用两层结构
                                 dropout=rnn_dropout,
                                 bidirectional=rnn_bidirectional)
        #
        # self.Linear_merger_father_2 = nn.Linear(2, 1)
        # self.Linear_merger_father_3 = nn.Linear(3, 1)
        # self.Linear_merger_father_4 = nn.Linear(4, 1)

    def forward(self, input_sequences, input_lengths, first_state=None):
        words_embedding = self.Embedding_layer(input_sequences)  # 从词向量层找出每一个单词的词向量

        packed = pack_padded_sequence(words_embedding, input_lengths)  # 根据每一个输入文本的长度压紧词向量

        outputs, state = self.RNN_layer(packed, first_state)  # 送入RNN中计算

        outputs, _ = pad_packed_sequence(outputs)  # 解压还原

        outputs = outputs[:, :, :self.input_hidden_size] + \
                  outputs[:, :, self.input_hidden_size:]  # 正向和反向叠加

        return outputs, state

    def forward_fathers(self, input_sequences, input_lengths, concepts_states=None):
        """
        :param input_sequences: 输入每个父概念的变量
        :param input_lengths: 每个概念的父概念的长度
        :param concepts_states:
        :return: 对每个概念的多个父概念的编码，以及对应的每个父概念的长度
        """
        max_fathers_lengths = max(max(i) if i else 0 for i in input_lengths)  # 这一批次最长父概念长度
        all_fathers_outputs = []  # 用list来收集所有的父概念
        for index, fathers in enumerate(input_sequences):
            if len(input_lengths[index]) != 0:  # 如果当前概念有父概念
                if len(input_lengths[index]) == 1:  # 一个父概念
                    # 如果只有一个父概念，就直接送入RNN中计算
                    words_embedding = self.Embedding_layer(fathers.transpose(0, 1))
                    packed = pack_padded_sequence(words_embedding, input_lengths[index])
                    state = self.__get_own_states(concepts_states, index, 1)
                    outputs, state = self.RNN_layer(packed, state)
                    outputs, _ = pad_packed_sequence(outputs)
                    outputs = self.__overlap_outputs(outputs)  # T*B*H
                else:  # 两个以上的父概念
                    # 两个以上的父概念，按长度降序排列后送入RNN中计算
                    fathers_lengths = {}
                    for i in range(len(input_lengths[index])):
                        fathers_lengths[i] = input_lengths[index][i]
                    sort_fathers_lengths = sorted(fathers_lengths, key=lambda x: fathers_lengths[x],
                                                  reverse=True)
                    temp_tensor = Variable(torch.LongTensor(len(input_lengths[index]), max_fathers_lengths))
                    if param.use_gpu:
                        temp_tensor = temp_tensor.cuda()
                    for i in range(len(input_lengths[index])):
                        temp_tensor[i, :] = fathers[sort_fathers_lengths[i], :]

                    words_embedding = self.Embedding_layer(temp_tensor.transpose(0, 1))
                    packed = pack_padded_sequence(words_embedding, sorted(input_lengths[index], reverse=True))
                    state = self.__get_own_states(concepts_states, index, words_embedding.shape[1])
                    outputs, state = self.RNN_layer(packed, state)
                    outputs, _ = pad_packed_sequence(outputs)
                    outputs = self.__overlap_outputs(outputs)  # T*B*H

                all_fathers_outputs.append(outputs)  # 用list来收集每一个概念父概念的编码
            else:
                all_fathers_outputs.append(None)  # 如果没有父概念也要空出这个位置，便于后面识别计算

                #     if len(input_lengths[index]) == 2:
                #         outputs = self.Linear_merger_father_2(outputs.transpose(2, 1)).transpose(2, 1)
                #     elif len(input_lengths[index]) == 3:
                #         outputs = self.Linear_merger_father_3(outputs.transpose(2, 1)).transpose(2, 1)
                #     else:
                #         outputs = self.Linear_merger_father_4(outputs.transpose(2, 1)).transpose(2, 1)
                #
                # all_fathers_outputs[:outputs.shape[0], index, :] = outputs[:, 0, :]

        return all_fathers_outputs, input_lengths

    def __overlap_outputs(self, outputs):
        return outputs[:, :, :self.input_hidden_size] + outputs[:, :, self.input_hidden_size:]

    def __get_own_states(selfs, state, index, batch_size):
        new_state = (state[0][:, index, :].unsqueeze(1).contiguous(),
                     state[1][:, index, :].unsqueeze(1).contiguous())

        for i in range(batch_size-1):
            new_state = (torch.cat((new_state[0], new_state[0]), dim=1),
                         torch.cat((new_state[1], new_state[1]), dim=1))

        return new_state

    """
    def forward_fathers(self, input_sequences, input_lengths, concepts_states=None):
        max_fathers_lengths = max(max(i) if i else 0 for i in input_lengths)
        all_fathers_outputs = Variable(torch.zeros(param.batch_size, max_fathers_lengths,
                                                   self.input_hidden_size))
        for index, fathers in enumerate(input_sequences):
            if len(input_lengths[index]):  # 如果当前概念有父概念
                temp = Variable(torch.zeros(1, max_fathers_lengths, self.input_hidden_size))
                if param.use_gpu:
                    temp = temp.cuda()
                for index_1, father in enumerate(fathers):  # 循环每一个父概念
                    real_words = father[: input_lengths[index][index_1]]
                    words_embedding = self.Embedding_layer(real_words).unsqueeze(1)

                    state = (concepts_states[0][:, index, :].unsqueeze(1).contiguous(),
                             concepts_states[1][:, index, :].unsqueeze(1).contiguous())
                    outputs, state = self.RNN_layer(words_embedding, state)

                    outputs = outputs[:, :, :self.input_hidden_size] + \
                              outputs[:, :, self.input_hidden_size:]  # 正向和反向叠加

                    temp[:, : outputs.shape[1], :] += outputs[0]
                    # TODO 若干个文本向量按一般的处理方法是级联这些向量，然后再变换操作

                all_fathers_outputs[index] = temp[0]

        all_fathers_outputs = all_fathers_outputs.transpose(0, 1)

        return all_fathers_outputs, state
    """



# 以下是原始的编码器
class EncoderRNN(nn.Module): # RNN编码器
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0.1):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size # 输入序列词所含的词数量
        self.hidden_size = hidden_size # 隐藏层大小（也等于词向量层维度）
        self.n_layers = n_layers # GRU层数
        self.dropout = dropout # dropout大小

        self.embedding = nn.Embedding(input_size, hidden_size) # 构建二维的词向量层，大小为[input_size, hidden_size]
        # 构建RNN计算层，输入和输出大小都是hidden_size，一层，双向
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True)

    def forward(self, input_seqs, input_lengths, hidden=None):
        """
        :param input_seqs: [max_len, batch_size]max_len*batch_size，即每一列是一个实际序列，是序列最大长度*批次（序列个数）
        :param input_lengths: 大小为batch个list，指示每个输入序列（每一列）实际长度有多少，用于压紧序列操作
        :param hidden: 如果不指定hidden则为0
        :return:
        """
        # Note: we run this all at once (over multiple batches of multiple sequences)
        # 注意：我们一次全部运行（多批次的多个序列）
        # 一次性获取全部输入序列的字符的词向量，大小为[max_len, batch_size, hidden_size]
        embedded = self.embedding(input_seqs)
        """
        这个函数的设置是为了对原来从embedded出来的数据进行压紧（原来有冗余，也就是有的句子不满最大长度，使用空替代的，所有现在要压紧它）
        输入(input, Variable):[max_len, batch_size, hidden_size], 
            每个序列的长度(length,list[int], 降序排列)，
            批大小在前（batch_first=True，默认否）
        输入：PackedSequence对象
        """
        # embedded中每一个序列里，序列不满最大长度填充为0，压紧操作
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        """
        这个函数的设置是为了对上述的压缩进行解压
        输入是压紧函数的对象
        输出是解压出来后的数据，是一个元祖，有两个数据
        一个是计算后解压的数据项，对应上个函数的输入[max_len, batch, hidden_size], 
        还有一个是数据是每个序列的长度[batch个每个最大长度的实际含有值]
        """
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)
        # 双向RNN的前后叠加
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs
        return outputs, hidden
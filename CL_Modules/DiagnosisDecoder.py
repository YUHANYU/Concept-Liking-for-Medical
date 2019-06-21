#-*-coding:utf-8-*-

# 诊断解码器

import torch
from torch import nn

from CL_Modules.Attention import Attn
from CL_Modules.Attention import AttnFathers

from Parameters import Parameters
param = Parameters()


# 诊断解码器
class DiagnosisDecoder(nn.Module):

    def __init__(self, target_words_num, target_hidden_size, attention_model='dot', rnn_layers=2,
                 rnn_dropout=0):
        super(DiagnosisDecoder, self).__init__()

        self.target_words_num = target_words_num
        self.target_hidden_size = target_hidden_size
        self.rnn_layers = rnn_layers

        self.Embedding_layer = nn.Embedding(target_words_num, target_hidden_size)
        # TODO 载入预训练好的词向量

        self.Embedding_Drooput_layer = nn.Dropout(rnn_dropout)

        self.RNN_layer = nn.LSTM(input_size=target_hidden_size,
                                 hidden_size=target_hidden_size,
                                 num_layers=rnn_layers,
                                 dropout=rnn_dropout)

        self.Linear_concat = nn.Linear(target_hidden_size * 2, target_hidden_size)

        self.Linear_out = nn.Linear(target_hidden_size, target_words_num)

        if attention_model:
            self.Attention_layer = Attn(attention_model, target_hidden_size)
            self.Attention_layer_fathers = AttnFathers(attention_model, target_hidden_size)

        self.softmax_1 = nn.Softmax(dim=0)
        self.softmax_2 = nn.Softmax(dim=1)

    def forward(self, target_sequences, last_state, encoder_outputs):
        batch_size = target_sequences.size(0)  # 批次大小
        words_embedding = self.Embedding_layer(target_sequences)  # 获取每一个单词的词向量
        words_embedded = self.Embedding_Drooput_layer(words_embedding)  # 对词向量做dropout处理
        words_embedded = words_embedded.view(1, batch_size, self.target_hidden_size)  # 大小变换

        outputs, state = self.RNN_layer(words_embedded, last_state)  # 将词向量和隐藏状态送入RNN中计算

        # 计算RNN隐藏层的输出和encoder_outputs的得分，即attention机制
        attention_weights = self.Attention_layer(outputs, encoder_outputs)
        context = attention_weights.bmm(encoder_outputs.transpose(0, 1))

        # Luong 公式 5, 级联RNN输出和上下文向量context
        concat_input = torch.cat((outputs.squeeze(0), context.squeeze(1)), 1)
        concat_output = torch.tanh(self.Linear_concat(concat_input))

        # Luong 公式 6，生成当前字符的概率，没有使用softmax
        outputs = self.Linear_out(concat_output)

        outputs_1 = self.softmax_1(outputs)
        outputs_2 = self.softmax_2(outputs)

        return outputs, state

    # =================================================================================
    # TODO 在诊断解码器中，还需要融合父概念（如果有），和病人的特征信息（当前是文本，以后换为数值）
    # ==================================================================================
    def merge_fathers_features(self, target_sequences, last_state, encoder_outputs,
                               father_outputs, fathers_lengths, features_sequences=None):
        batch_size = target_sequences.size(0)  # 批次大小
        words_embedding = self.Embedding_layer(target_sequences)  # 获取每一个单词的词向量
        words_embedded = self.Embedding_Drooput_layer(words_embedding)  # 对词向量做dropout处理
        words_embedded = words_embedded.view(1, batch_size, self.target_hidden_size)  # 大小变换

        outputs, state = self.RNN_layer(words_embedded, last_state)  # 将词向量和隐藏状态送入RNN中计算

        # 计算RNN隐藏层的输出和encoder_outputs的得分，即attention机制
        attention_weights = self.Attention_layer(outputs, encoder_outputs)
        context = attention_weights.bmm(encoder_outputs.transpose(0, 1))

        # 计算RNN隐藏层的输出和father_encoder_outputs的得分
        attention_weights_fathers = self.Attention_layer_fathers(outputs, father_outputs, fathers_lengths)
        batch_size = len(fathers_lengths)
        context_fathers = [None for i in range(batch_size)]
        for i in range(batch_size):
            if len(fathers_lengths[i]) != 0:
                temp = attention_weights_fathers[i].unsqueeze(1).permute(2, 1, 0).\
                    bmm(father_outputs[i].permute(1, 0, 2))
                context_fathers[i] = temp

        outputs_context_context_fathers_concat = 0
        for i in range(batch_size):
            if len(fathers_lengths) != 0:
                pass



# 以下是原始的解码器
class LuongAttnDecoderRNN(nn.Module):

    def __init__(self, attn_model, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_seq, last_hidden, encoder_outputs):
        """
        :param input_seq: 当前的输入标记，[batch个字符索引]
        :param last_hidden: 上一个时间步隐藏状态[_, B, H], _*max_len*hidden_size
        :param encoder_outputs: 编码器的全部输出[T, B, H], （编码器序列的）max_len*batch_size*hidden_size
        :return:
        """
        # Note: we run this one step at a time

        # Get the embedding of the current input word (last output word)
        batch_size = input_seq.size(0)
        embedded = self.embedding(input_seq)
        embedded = self.embedding_dropout(embedded)
        embedded = embedded.view(1, batch_size, self.hidden_size)  # S=1 x B x N

        # Get current hidden state from input word and last hidden state
        rnn_output, hidden = self.gru(embedded, last_hidden)

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        attn_weights = self.attn(rnn_output, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # B x S=1 x N

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        rnn_output = rnn_output.squeeze(0)  # S=1 x B x N -> B x N
        context = context.squeeze(1)  # B x S=1 x N -> B x N
        concat_input = torch.cat((rnn_output, context), 1)
        # concat_output = F.tanh(self.concat(concat_input)) 上面的F.tanh已经被弃用了
        concat_output = torch.tanh(self.concat(concat_input))

        # Finally predict next token (Luong eq. 6, without softmax)
        output = self.out(concat_output)

        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, attn_weights
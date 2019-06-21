#-*-coding:utf-8-*-

# 徐磊到序列的模型，主要执行训练，验证和测试的过程

import random

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
torch.backends.cudnn.benchmark = True

from Parameters import Parameters
param = Parameters()


# 概念生成诊断的生成类
class Concept2Diagnosis(object):

    def __init__(self, concepts_encoder, features_encoder, diagnosis_decoder):
        self.concepts_encoder = concepts_encoder
        self.features_encoder = features_encoder
        self.diagnosis_decoder = diagnosis_decoder

        self.concepts_encoder_optimizer, self.optimizer_name = param.set_optimizer\
            (concepts_encoder, param.learning_rate)
        self.features_encoder_optimizer, _ = param.set_optimizer(features_encoder, param.learning_rate)
        self.diagnosis_decoder_optimizer,_ = param.set_optimizer(diagnosis_decoder, param.learning_rate)

        self.criterion = nn.CrossEntropyLoss()

    def __sequence_mask(self, sequence_length, max_len=None):
        if max_len is None:
            max_len = sequence_length.data.max()
        batch_size = sequence_length.size(0)

        # seq_range = torch.range(0, max_len - 1).long()
        seq_range = torch.arange(0, max_len).long()

        seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
        seq_range_expand = Variable(seq_range_expand)
        if sequence_length.is_cuda:
            seq_range_expand = seq_range_expand.cuda()
        seq_length_expand = (sequence_length.unsqueeze(1)
                             .expand_as(seq_range_expand))
        return seq_range_expand < seq_length_expand

    def __masked_cross_entropy(self, logits, target, length):
        length = Variable(torch.LongTensor(length)).cuda()

        """
        Args:
            logits: A Variable containing a FloatTensor of size
                (batch, max_len, num_classes) which contains the
                unnormalized probability for each class.
            target: A Variable containing a LongTensor of size
                (batch, max_len) which contains the index of the true
                class for each corresponding step.
            length: A Variable containing a LongTensor of size (batch,)
                which contains the length of each data in a batch.
        Returns:
            loss: An average loss value masked by the length.
        """

        # logits_flat: (batch * max_len, num_classes)
        logits_flat = logits.view(-1, logits.size(-1))
        # log_probs_flat: (batch * max_len, num_classes)
        # log_probs_flat = functional.log_softmax(logits_flat)
        log_probs_flat = F.log_softmax(logits_flat, dim=1)
        # target_flat: (batch * max_len, 1)
        target_flat = target.view(-1, 1)
        # losses_flat: (batch * max_len, 1)
        losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
        # losses: (batch, max_len)
        losses = losses_flat.view(*target.size())
        # mask: (batch, max_len)
        mask = self.__sequence_mask(sequence_length=length, max_len=target.size(1))
        losses = losses * mask.float()
        loss = losses.sum() / length.float().sum()
        return loss

    def concepts_features_2_diagnosis(
            self, input_concepts_batches, input_concepts_lenths,input_fathers_concepts_batches,
            input_fathers_concepts_lengths, target_diagnosis_batches, target_lengths):
        batch_size = input_concepts_batches.shape[1]  # 获取批大小

        self.concepts_encoder_optimizer.zero_grad()
        # self.features_encoder_optimizer.zero_grad()
        self.diagnosis_decoder_optimizer.zero_grad()

        encoder_outputs, encoder_state = self.concepts_encoder(
            input_concepts_batches, input_concepts_lenths, None)

        # 如果概念有父概念，还需要编码父概念信息
        if param.use_fathers_concepts:
            fathers_outputs, fathers_lengths = self.concepts_encoder.forward_fathers(
                input_fathers_concepts_batches, input_fathers_concepts_lengths, encoder_state)

        decoder_input = Variable(torch.LongTensor([param.SOS_token] * batch_size))  # 解码器的初始输入是SOS开始符
        # 解码器的初始状态=编码器最后时刻的状态
        decoder_hidden = (encoder_state[0][:self.diagnosis_decoder.rnn_layers],
                          encoder_state[1][:self.diagnosis_decoder.rnn_layers])

        max_target_length = max(target_lengths)
        all_decoder_outputs = Variable(torch.zeros(max_target_length, batch_size,
                                                   self.diagnosis_decoder.target_words_num))

        if param.use_gpu:
            decoder_input = decoder_input.cuda()
            all_decoder_outputs = all_decoder_outputs.cuda()

        for t in range(max_target_length):
            if not param.use_fathers_concepts:
                decoder_output, decoder_hidden = self.diagnosis_decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
            else:
                self.diagnosis_decoder.merge_fathers_features(
                    decoder_input, decoder_hidden, encoder_outputs, fathers_outputs, fathers_lengths)
            all_decoder_outputs[t] = decoder_output

            use_teacher_forcing = True if random.random() < param.teacher_forcing_ratio else False

            if use_teacher_forcing:
                top_values, top_indexes = decoder_output.data.topk(1)
                ni = [int(top_indexes[i][0].cpu().numpy()) for i in range(top_indexes.shape[0])]
                decoder_input = Variable(torch.LongTensor(ni))
                if param.use_gpu:
                    decoder_input = decoder_input.cuda()
            else:
                decoder_input = target_diagnosis_batches[t]

        loss = self.__masked_cross_entropy(
            all_decoder_outputs.transpose(0, 1).contiguous(),
            target_diagnosis_batches.transpose(0, 1).contiguous(),
            target_lengths
        )
        loss.backward()

        clip_value = 20.0
        concepts_encoder_clip = clip_grad_norm_(self.concepts_encoder.parameters(), clip_value)
        # features_encoder_clip = clip_grad_norm_(self.features_encoder.parameters(), clip_value)
        diagnosis_decoder_clip = clip_grad_norm_(self.diagnosis_decoder.parameters(), clip_value)

        self.concepts_encoder_optimizer.step()
        # self.features_encoder_optimizer.step()
        self.diagnosis_decoder_optimizer.step()

        return loss.item()


# 原始训练代码
lab_num = 10
def train(input_batches, input_lengths, target_batches, target_lengths, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion, max_length=lab_num):
    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0  # Added onto for each word

    # Run words through encoder 这里是编码器批概念信息
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)

    # Prepare input and output variables
    decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))
    decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder

    max_target_length = max(target_lengths)
    all_decoder_outputs = Variable(torch.zeros(max_target_length, bat, decoder.output_size))

    # Move new Variables to CUDA
    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        all_decoder_outputs = all_decoder_outputs.cuda()

    # Run through decoder one time step at a time
    for t in range(max_target_length):
        decoder_output, decoder_hidden, decoder_attn = decoder(decoder_input, decoder_hidden, encoder_outputs)

        all_decoder_outputs[t] = decoder_output
        # 下一个输入使用的当前实际的目标，即没有使用教师强制训练机制，能更快收敛，但可能造成测试效果不好
        # decoder_input = target_batches[t]  # Next input is current target
        # print(len(decoder_input), decoder_input)
        # exit()

        #################### 这里改为交替使用教师强制训练机制 #################################
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        if use_teacher_forcing: # 如果使用教师强制训练机制
            topv, topi = decoder_output.data.topk(1)
            ni = [int(topi[i][0].cpu().numpy()) for i in range(topi.shape[0])] # 存在着优化的方法
            decoder_input = Variable(torch.LongTensor(ni)) # 下一个输入的词是当前预测出来的词
            if USE_CUDA: decoder_input = decoder_input.cuda()
        else: # 不使用教师强制训练机制
            decoder_input = target_batches[t]

    # Loss calculation and backpropagation
    loss = masked_cross_entropy(
        all_decoder_outputs.transpose(0, 1).contiguous(),  # -> batch x seq
        target_batches.transpose(0, 1).contiguous(),  # -> batch x seq
        target_lengths
    )
    loss.backward()

    # Clip gradient norms
    ec = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    dc = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Update parameters with optimizers
    encoder_optimizer.step()
    decoder_optimizer.step()

    # return loss.data[0], ec, dc
    return loss.item(), ec, dc
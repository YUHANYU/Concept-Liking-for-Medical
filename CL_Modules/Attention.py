#-*-coding:utf-8-*-

# attention注意力机制

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from Parameters import Parameters
param = Parameters()



class Attn(nn.Module):

    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()

        self.method = method # 选择不同的score计算方式
        self.hidden_size = hidden_size # 隐藏层大小

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        """
        :param hidden: [max_len, batch_size]
        :param encoder_outputs: [max_len, batch_size, hidden_size]
        :return:
        """
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)

        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(this_batch_size, max_len))  # B x S

        if param.use_gpu:
            attn_energies = attn_energies.cuda()

        # For each batch of encoder outputs
        for b in range(this_batch_size):
            # Calculate energy for each encoder output
            for i in range(max_len):
                attn_energies[b, i] = self.score(hidden[:, b], encoder_outputs[i, b].unsqueeze(0))

        # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_output):

        if self.method == 'dot':
            energy = hidden.squeeze(0).dot(encoder_output.squeeze(0))
            return energy

        elif self.method == 'general':
            energy = self.attn(encoder_output).squeeze(0)
            energy = hidden.squeeze(0).dot(energy)
            return energy

        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1)).squeeze(0)
            energy = self.v.squeeze(0).dot(energy)
            return energy


# 把fathers概念融入的attention
class AttnFathers(nn.Module):

    def __init__(self, method, hidden_size):
        super(AttnFathers, self).__init__()

        self.method = method  # 选择不同的score计算方式
        self.hidden_size = hidden_size  # 隐藏层大小

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, fathers_outputs, fathers_lengths):
        this_batch_size = hidden.shape[1]
        scores = []

        for i in range(this_batch_size):
            if len(fathers_lengths[i]) != 0:  # 如果当前概念有父概念
                score = self.score(hidden[:, i], fathers_outputs[i])
                scores.append(score)
            else:
                scores.append(None)

        return scores

    def score(self, hidden, father_output):
        max_len = father_output.shape[0]  # 最大长度
        father_num = father_output.shape[1]  # 父亲概念的数量
        attn_energy = Variable(torch.zeros(max_len, father_num))
        if param.use_gpu:
            attn_energy = attn_energy.cuda()

        for i in range(father_num):
            for j in range(max_len):
                if self.method == 'dot':
                    attn_energy[j][i] = hidden.squeeze(0).dot(father_output[j][i])

                # TODO general的attention计算方式存在问题，需要调整，暂时无法使用
                elif self.method == 'general':
                    energy = self.attn(father_output[j][i])
                    attn_energy[j][i] = hidden.squeeze(0).dot(energy)

                # TODO concat的attention计算方式存在问题，需要调整，暂时无法使用
                elif self.method == 'concat':
                    energy = self.attn(torch.cat((hidden, father_output[j][i]), 1))
                    attn_energy[j][i] = self.v.squeeze(0).dot(energy)

        return F.softmax(attn_energy, dim=0)





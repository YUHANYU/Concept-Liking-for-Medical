#-*-coding:utf-8-*-

# 参数类

import os

import torch
from torch import optim


class Parameters():

    def __init__(self):

        self.batch_size = 64  # 128 256 512
        self.learning_rate = 0.00005
        self.train_epochs = 10
        self.hidden_size = 256
        self.use_gpu = True if torch.cuda.is_available() else False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.file_path = os.getcwd() + '/CL_Data/concept_diagnosis_features_fathers.txt'

        self.MIN_LENGTH = 0
        self.MAX_LENGTH = 30
        self.PAD_token = 0
        self.SOS_token = 1
        self.EOS_token = 2

        self.teacher_forcing_ratio = 0.5

        self.use_fathers_concepts = True
        self.use_features = True

    def set_optimizer(self, object, lr):
        self.optimizer = optim.Adam(object.parameters(), lr)
        self.optimizer_name = str(self.optimizer).split(' ')[0]

        return self.optimizer, self.optimizer_name
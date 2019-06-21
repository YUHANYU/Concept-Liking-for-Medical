#-*-coding:utf-8-*-

# 从概念（父概念）+ 特征信息到诊断的生成模型

import torch
from torch.utils.data import DataLoader

from CL_Modules.ConceptsEncoder import ConceptsEncoder
from CL_Modules.FeaturesEncoder import FeaturesEncoder
from CL_Modules.DiagnosisDecoder import DiagnosisDecoder
from CL_Modules.Concept2Diagnosis import Concept2Diagnosis

from CL_Tools.Data_Prepare import Data_Prepare
from CL_Tools.Others import train_val_test_1, show_loss

import time

from Parameters import Parameters
param = Parameters()


class ConceptsFeatures2Diagnosis(object):

    def __init__(self):
        self.__get_data()
        self.__build_encoder_decoder()

    def __get_data(self):
        data_obj = Data_Prepare()  # 数据对象
        self.data_obj = data_obj
        # 获取原始数据内容，组织好的数据内容，以及概念，诊断，特征三个字符类
        data_content, all_concept_fathers_diagnsis_features, \
        concepts_lang, diagnosis_lang, features_lang = data_obj.get_concept_diagnosis_fathers_fetures(
            param.file_path)

        self.concepts_words = concepts_lang.n_words  # 概念单词数
        self.features_words = features_lang.n_words  # 特征单词数
        self.diagnosis_words = diagnosis_lang.n_words  # 诊断单词数

        # 切分训练数据，验证数据和测试数据
        self.train_data, \
        self.val_data, \
        self.test_data = train_val_test_1(all_concept_fathers_diagnsis_features)

    def __build_encoder_decoder(self):
        self.concepts_encoder = ConceptsEncoder(
            input_words_num=self.concepts_words, input_hidden_size=param.hidden_size
        ).to(param.device)  # 概念编码器

        self.features_encoder = FeaturesEncoder(
            input_words_num=self.features_words, input_hidden_size=param.hidden_size
        ).to(param.device)  # 特征编码器

        self.diagnosis_decoder = DiagnosisDecoder(
            target_words_num=self.diagnosis_words, target_hidden_size=param.hidden_size
        ).to(param.device)  # 诊断解码器

    def train(self):
        data_loader = DataLoader(dataset=self.train_data, batch_size=param.batch_size, shuffle=True,
                                 drop_last=False)

        con_fea_dia = Concept2Diagnosis(concepts_encoder=self.concepts_encoder,
                                        features_encoder=self.features_encoder,
                                        diagnosis_decoder=self.diagnosis_decoder)

        show_y = []
        for one_epoch in range(param.train_epochs):
            for step, batch_data in enumerate(data_loader):
                concept_fathers = batch_data[0]
                diagnosis = batch_data[1]
                features = batch_data[2]

                concept_variable, concept_lengths, \
                diagnosis_variable, diagnosis_lengths, \
                features_variable, features_lengths, \
                fathers_variable, fathers_lengths = \
                    self.data_obj.word_2_tensor(concept_fathers, diagnosis, features)

                loss = con_fea_dia.concepts_features_2_diagnosis(
                    input_concepts_batches=concept_variable,
                    input_concepts_lenths=concept_lengths,
                    input_fathers_concepts_batches=fathers_variable,
                    input_fathers_concepts_lengths=fathers_lengths,
                    target_diagnosis_batches=diagnosis_variable,
                    target_lengths=diagnosis_lengths
                )
                show_y.append(round(loss, 4))

                print('Epoch: %3.0f' % one_epoch,
                      '| Step: %5.0f' % step,
                      '| Batch size: %4.0f' % len(batch_data[0]),
                      '| Loss: %6.4f' % round(loss, 4))

        picture_name = str(param.train_epochs) + '_' + \
                       str(param.batch_size) + '_' + \
                       str(con_fea_dia.optimizer_name) + "-" + \
                       str(param.learning_rate) + '_' + \
                       str(time.strftime("%m_%d_%H_%M_%S", time.localtime()))

        show_x = [i for i in range(len(show_y))]
        show_loss(show_x, show_y, picture_name)


def main():
    cf_2_d = ConceptsFeatures2Diagnosis()
    cf_2_d.train()


if __name__ == '__main__':
    main()
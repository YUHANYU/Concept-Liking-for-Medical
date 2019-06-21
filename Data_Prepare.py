# 数据组织类

#-*-coding:utf-8-*-

import re
import unicodedata

import torch
from torch.autograd import Variable

from Parameters import Parameters
param = Parameters()


# 字符统计类
class Lang:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"}
        self.n_words = 3 # 初始字典有3个字符

    def index_words(self, sentence):
        for word in sentence.split(' '):
            self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed: return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words %s / %s = %.4f' % (
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"}
        self.n_words = 3  # Count default tokens

        for word in keep_words:
            self.index_word(word)


class Data_Prepare(object):

    def __init__(self):
        self.concepts_lang = None
        self.diagnosis_lang = None
        self.features_lang = None

    def normalize_string(self, s):
        s = self.unicode_to_ascii(s.lower().strip())
        s = re.sub(r"([,.!?])", r" \1 ", s)
        s = re.sub(r"[^a-zA-Z,.!?]+", r" ", s)
        s = re.sub(r"\s+", r" ", s).strip()
        return s

    def unicode_to_ascii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    def filter_pairs(self, pairs):
        filtered_pairs = []
        for pair in pairs:
            sentence_num = 0
            for i in pair:
                if len(i.split(' ')) > param.MIN_LENGTH and len(i.split(' ')) <= param.MAX_LENGTH:
                    sentence_num += 1
            if sentence_num == len(pair):
                filtered_pairs.append(pair)
            else:
                temp = [len(i.split(' ')) for i in pair]
                print(temp, pair)

        return filtered_pairs

    def indexes_from_sentence(self, lang, sentence):
        return [lang.word2index[word] for word in sentence.split(' ')] + [param.EOS_token]

    def pad_seq(self, seq, max_length):
        seq += [param.PAD_token for i in range(max_length - len(seq))]
        return seq

    def get_concept_diagnosis_fathers_fetures(self, file_path:str):
        """
        :param file_path:
        :return:
        """
        data_content = open(file_path).read().strip().split('\n')  # 读出内容
        new_data_content_1 = [[self.normalize_string(s)
                             for s in l.split('\t')]
                            for l in data_content] # 规范化字符
        new_data_content_2 = self.filter_pairs(new_data_content_1)  # 过滤较长的句子

        diagnosis_lang = Lang('diagnosis')
        features_lang = Lang('features')
        concepts_lang = Lang('concept')

        self.diagnosis_lang = diagnosis_lang
        self.concepts_lang = concepts_lang
        self.features_lang = features_lang

        all_conceptFathers_diagnsis_features = []  # 重新组织的后的数据

        for line in new_data_content_2:
            concept = line[0]  # 概念文本
            diagnosis = line[1]  # 诊断文本
            features = line[2]  # 特征文本

            # 检索文本中所涉及的各个单词
            diagnosis_lang.index_words(diagnosis)
            features_lang.index_words(features)
            concepts_lang.index_words(concept)
            for i in range(3, len(line)):
                concepts_lang.index_words(line[i])

            # 把概念和父概念组织起来
            concept_fathers = concept
            for i in range(3, len(line)):
                concept_fathers += '\t'
                concept_fathers += line[i]

            a_conceptFathers_diagnsis_features = [concept_fathers, diagnosis, features]
            all_conceptFathers_diagnsis_features.append(a_conceptFathers_diagnsis_features)

        # print('在概念-父概念输入序列中的单词数为：', concepts_lang.n_words)
        # print('在个人信息输入序列中的单词数为：', features_lang.n_words)
        # print('在诊断目标序列中的单词数为：', diagnosis_lang.n_words)

        return new_data_content_2, all_conceptFathers_diagnsis_features, \
               concepts_lang, diagnosis_lang, features_lang

    def word_2_tensor(self, conceptFathers, diagnosis, features):
        """
        把单词转化为tensor，送入编码器和解码器中计算
        :param conceptFthers:
        :param diagnosis:
        :param features:
        :return:
        """
        concept_sequence = []
        diagnosis_sequence = []
        features_sequence = []
        fathers_sequence = []

        for i in range(len(features)):
            concept_sequence.append(self.indexes_from_sentence(
                self.concepts_lang, conceptFathers[i].split('\t')[0]))
            diagnosis_sequence.append(self.indexes_from_sentence(
                self.diagnosis_lang, diagnosis[i]))
            features_sequence.append(self.indexes_from_sentence(
                self.features_lang, features[i]))
            fathers = []
            for j in range(1, len(conceptFathers[i].split('\t'))):
                fathers.append(self.indexes_from_sentence(
                    self.concepts_lang, conceptFathers[i].split('\t')[j]))
            fathers_sequence.append(fathers)

        seq_pairs = sorted(zip(concept_sequence, diagnosis_sequence, features_sequence, fathers_sequence),
                           key=lambda p: len(p[0]), reverse=True)
        concept_sequence, diagnosis_sequence, features_sequence, fathers_sequence = zip(*seq_pairs)

        concept_lengths = [len(s) for s in concept_sequence]
        concept_padded = [self.pad_seq(s, max(concept_lengths)) for s in concept_sequence]

        diagnosis_lengths = [len(s) for s in diagnosis_sequence]
        diagnosis_padded = [self.pad_seq(s, max(diagnosis_lengths)) for s in diagnosis_sequence]

        features_lengths = [len(s) for s in features_sequence]
        features_padded = [self.pad_seq(s, max(features_lengths)) for s in features_sequence]

        # 注意fathers_lengths中可能会出现null的情况，即就是概念为子概念，没有父概念
        fathers_lengths = [[len(i) for i in s] for s in fathers_sequence]
        fathers_length_max = max([max(i) if i else 0 for i in fathers_lengths])
        fathers_padded = [[self.pad_seq(i, fathers_length_max) for i in s]
                          if s else 0 for s in fathers_sequence]

        diagnosis_variable = Variable(torch.LongTensor(diagnosis_padded)).transpose(0, 1)
        concept_variable = Variable(torch.LongTensor(concept_padded)).transpose(0, 1)
        features_variable = Variable(torch.LongTensor(features_padded)).transpose(0, 1)

        if param.use_gpu:
            diagnosis_variable = diagnosis_variable.cuda()
            concept_variable = concept_variable.cuda()
            features_variable = features_variable.cuda()
            fathers_variable = [Variable(torch.LongTensor(i)).transpose(0, 1).cuda() if i else None
                               for i in fathers_padded]
        else:
            fathers_variable = [Variable(torch.LongTensor(i)).transpose(0, 1) if i else None
                                for i in fathers_padded]

        return concept_variable, concept_lengths, diagnosis_variable, diagnosis_lengths,\
               features_variable, features_lengths, fathers_variable, fathers_lengths


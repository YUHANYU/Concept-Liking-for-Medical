#-*-coding:utf-8-*-

#=============================#
# tensor变换


# import torch
# from torch.autograd import Variable
#
# a = Variable(torch.LongTensor([[15, 98, 789, 1, 0, 0],
#                                [15, 94, 847, 0, 0, 0]]))
# row, col = a.shape
# print(a, a.shape)
#
# b = Variable(torch.LongTensor(row, col))
# b[0, :] = a[1, :]
# b[1, :] = a[0, :]
# print(b, b.shape)


#===============================#
# 三维的tensor能否放到list中


import torch

a = torch.LongTensor(2, 1, 2)
print(a)

print(a[0].shape)
print(a[0])
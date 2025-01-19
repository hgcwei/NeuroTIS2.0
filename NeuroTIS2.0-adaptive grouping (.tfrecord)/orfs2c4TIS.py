import numpy as np
import string
import random

class ORFs2C4TIS:

    # seq_ls（包含多个序列的列表）、ORFs_loc_lsls（包含多个ORF的位置信息列表的列表）、wds（窗口大小）
    def __init__(self,seq_ls, ORFs_loc_lsls,wds):

        self.seq_ls = seq_ls
        self.ORFs_loc_lsls = ORFs_loc_lsls
        self.wds = wds
        self.s = int(wds/2)
    # 遍历每个序列的ORF位置信息列表，对于每个ORF位置信息，调用 gen_c4_for_tis 方法生成与TIS相关的序列。
    # 然后，调用 gen_c4_for_seq 方法将生成的序列转换为特征表示，并将结果存储在 c4_fea_lsls 列表中。
    # 返回最终的特征列表 c4_fea_lsls。
    def gen_c4_for_tiss(self):

        c4_fea_lsls = []
        for i in range(len(self.ORFs_loc_lsls)):

            seq = self.seq_ls[i]
            ORFs_loc_ls = self.ORFs_loc_lsls[i]
            c4_fea_ls = []
            for j in range(len(ORFs_loc_ls)):
                ORFs_loc = ORFs_loc_ls[j]
                start = ORFs_loc[0]
                c4 = self.gen_c4_for_tis(seq,start)
                c4_fea = self.gen_c4_for_seq(c4)
                c4_fea_ls.append(c4_fea)
            c4_fea_lsls.append(c4_fea_ls)

        return c4_fea_lsls
    # 接受两个参数，seq 表示当前序列，start 表示ORF的起始位置。
    # 根据起始位置 start 在序列 seq 中生成长度为 wds 的序列 c4。
    # 如果 start 小于0或大于序列长度-1，将对应位置设置为 'N'（表示未知）。
    # 返回生成的序列 c4
    def gen_c4_for_tis(self,seq,start):

        c4 = []
        for i in range(start-self.s,start+self.s,1):
            if i < 0 or i > (len(seq)-1):
                c4.append('N')
            else:
                c4.append(seq[i])

        return ''.join(c4)

    # 将输入的序列 seq 转换为特征表示，其中每个碱基用一个长度为4的二进制数组表示。
    # 'A' 被表示为 [1,0,0,0]、'C' 为 [0,1,0,0]、'G' 为 [0,0,1,0]、'T' 为 [0,0,0,1]，未知碱基为 [0,0,0,0]。
    # 返回表示序列特征的二维数组 data
    def gen_c4_for_seq(self,seq):
        data = np.zeros((len(seq),4),dtype=np.uint8)
        for i in range(len(seq)):
            if seq[i] == 'A':
                data[i] = [1,0,0,0]
            elif seq[i] == 'C':
                data[i] = [0,1,0,0]
            elif seq[i] == 'G':
                data[i] = [0,0,1,0]
            elif seq[i] == 'T':
                data[i]= [0,0,0,1]
            else:
                data[i] = [0,0,0,0]
        return data

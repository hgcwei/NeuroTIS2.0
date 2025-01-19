
import math
import numpy as np
from tcn import compiled_tcn
import cuRNA
import pandas as pd
import csv


class RNAs2cod:

    def __init__(self,seq_ls,model_weights):

        self.model = compiled_tcn(num_feat=65,
                             num_classes=2,
                             nb_filters=15,
                             kernel_size=20,
                             dilations=[1, 2, 4],
                             nb_stacks=2,
                             max_len=2000,
                             use_skip_connections=False,
                             opt='rmsprop',
                             padding='same',
                             lr=1e-3,
                             use_weight_norm=True,
                             return_sequences=True)

        self.model.load_weights(model_weights)
        self.cu = cuRNA.CuRNA(['A', 'C', 'G', 'T'], 90)
        self.seq_ls = seq_ls
        self.n = len(seq_ls)

    # 为输入的RNA序列生成密码子使用率（codon usage）的矩阵。
    # 遍历每个RNA序列，获取三个阅读框（0, 1, 2）上的ORF（开放阅读框）。
    # 对每个ORF，获取滑动窗口内的密码子使用率矩阵。
    # 将三个阅读框的密码子使用率矩阵垂直堆叠，形成一个大的矩阵。
    # 返回生成的密码子使用率矩阵和一个包含三个元组的列表，每个元组包含对应的阅读框的密码子使用率矩阵的行数。
    def gen_codon_usage_for_test(self):

        mat_cus = []
        wid_s = []

        for i in range(self.n):
            rna = self.seq_ls[i]

            orf0 = self.cu.get_orf(rna, 0)
            orf1 = self.cu.get_orf(rna, 1)
            orf2 = self.cu.get_orf(rna, 2)

            mat_cu0 = self.cu.get_slid_cu(orf0)
            mat_cu1 = self.cu.get_slid_cu(orf1)
            mat_cu2 = self.cu.get_slid_cu(orf2)

            wid_s.append((mat_cu0.shape[0], mat_cu1.shape[0], mat_cu2.shape[0]))
            mat_cus.append(np.vstack((mat_cu0, mat_cu1, mat_cu2)))

        return np.vstack((mat_cus)),wid_s

    def gen_coding_scores_for_rnas(self):
        # 调用 gen_codon_usage_for_test 方法获取密码子使用率矩阵和阅读框行数信息。
        mat, wid = self.gen_codon_usage_for_test()
        l_rna = mat.shape[0]
        # 对密码子使用率矩阵进行处理，将其变形为三维数组（按每1000个为一组）。
        num = math.floor(l_rna / 2000) * 2000
        if num == l_rna:
            data = mat[0:num, :]
        else:
            data = np.vstack((mat[0:num, :], mat[-2000:, :]))
        data = data.reshape((-1, 2000, 65))
        # 使用 TCN 模型对处理后的数据进行预测，得到编码得分。
        scores = self.model.predict(data)
        # 调用 compute_scores_for_rna 方法处理得分，返回最终的编码得分。
        codings_scores_rna = self.compute_scores_for_rna(scores, l_rna, num, wid)

        return codings_scores_rna

    # 将 TCN 模型的预测得分按照阅读框的行数信息进行分组。
    # 返回一个列表，其中每个元素是一个元组，包含三个列表，分别对应三个阅读框的得分。
    def compute_scores_for_rna(self,scores,l_seq,n,wid):

        rna_scores = []
        scores = scores[:, :, 1]
        scores = np.reshape(scores, [-1, 1])
        scores = scores.tolist()

        if l_seq != n:
            del scores[l_seq - 1000:n]

        for i in range(len(wid)):
            s1 = scores[0: wid[i][0]]
            del scores[0: wid[i][0]]
            s2 = scores[0: wid[i][1]]
            del scores[0: wid[i][1]]
            s3 = scores[0: wid[i][2]]
            del scores[0: wid[i][2]]
            rna_scores.append((s1,s2,s3))



            # with open('cs_ls_test.csv', 'a', newline='') as file:
            #     writer = csv.writer(file)
            #     writer.writerows(rna_scores)
            # mean_f0 = np.mean(s1)
            # mean_f1 = np.mean(s2)
            # mean_f2 = np.mean(s3)
            # max_c = np.max((mean_f0,mean_f1,mean_f2))


        return rna_scores

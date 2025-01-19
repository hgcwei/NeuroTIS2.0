import cuRNA
from tcn import compiled_tcn
import numpy as np


class ORFs2CodFea:

    # 接受四个参数：seq_ls（包含多个序列的列表）、ORF_loc_lsls（包含多个 ORF 位置信息列表的列表）、
    # wds（窗口大小）、cs_ls（可能是 codon usage 的列表）。
    # 初始化类的属性，包括序列列表 seq_ls、ORF 位置信息列表的列表 ORF_loc_lsls、
    # 序列数量 n、cs 列表 cs_ls 和窗口大小 wds
    def __init__(self,seq_ls, ORF_loc_lsls, wds,cs_ls):


        self.seq_ls = seq_ls

        self.ORF_loc_lsls = ORF_loc_lsls

        self.n = len(seq_ls)

        self.cs_ls = cs_ls

        self.wds = wds
    # 遍历 ORF 位置信息列表的列表，对于每个 ORF 的三个阅读框的 cs 列表，取最小长度 m。
    # 将三个阅读框的 cs 列表合并成一个列表 cs。
    # 遍历每个 ORF 的位置信息，调用 gen_glob_cs_fea 方法生成全局 cs 特征，并将结果存储在 glob_cs_lsls 列表中。
    # 返回最终的全局 cs 特征列表 glob_cs_lsls。
    def gen_cs_for_tis_pred(self):

        glob_cs_lsls = []

        for i in range(len(self.ORF_loc_lsls)):

            cs0, cs1, cs2 = self.cs_ls[i]
            m = min(len(cs0), len(cs1), len(cs2))
            cs = []
            for k in range(m):
                cs.append(cs0[k][0])
                cs.append(cs1[k][0])
                cs.append(cs2[k][0])

            glob_cs_ls = []

            for j in range(len(self.ORF_loc_lsls[i])):

                ORF_loc = self.ORF_loc_lsls[i][j]

                glob_cs_ls.append(self.gen_glob_cs_fea(ORF_loc,cs))

            glob_cs_lsls.append(glob_cs_ls)

        return glob_cs_lsls
    # 接受两个参数，ORF_loc 表示当前 ORF 的位置信息，cs 表示 cs 列表。
    # 初始化一个长度为 (self.wds * 2) 的 cs 特征数组 cs_fea，并将其初始化为 0。
    # 计算半窗口大小 s，并获取 ORF 的起始位置 start 和结束位置 end。
    # 遍历 ORF 的起始和结束窗口，将 cs 列表中对应位置的值添加到 cs_fea 中。
    # 如果窗口位置超出 cs 列表的范围，则将 cs_fea 中对应位置的值设置为 -1。
    # 返回生成的 cs 特征数组 cs_fea
    def gen_glob_cs_fea(self,ORF_loc, cs):
        cs_fea= np.zeros((self.wds*2))
        s = int(self.wds/2)
        start = ORF_loc[0]
        end = ORF_loc[1]
        j = 0
        for i in range(start-s,start+s,1):
            if i < 0 or i > len(cs)-1:
                cs_fea[j] = -1
            else:
                cs_fea[j] = cs[i]
            j = j + 1

        for i in range(end-s,end+s,1):
            if i < 0 or i > len(cs)-1:
                cs_fea[j] = -1
            else:
                cs_fea[j]= cs[i]
            j = j +1

        return cs_fea




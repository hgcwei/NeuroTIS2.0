# **************************************************************************************************************************
#  Author: Chao Wei
#  Function: this code generates global features in paper Mixed Gaussian Model for TIS
#  Input:   seq_ls, ORF_loc_lsls
#  Output:  features list
#  paper:  NeruoTIS+: ...
# **************************************************************************************************************************

import re
import math
import fa2orfs
import numpy as np
# 接受两个参数，pattern 表示要查找的子字符串，string 表示待查找的字符串。
# 使用正则表达式找到字符串中所有匹配 pattern 的起始位置，并返回位置列表 ind_ls。
def getSubstrLs(pattern, string):
    ind_ls = []
    for match in re.finditer(pattern, string):
        ind_ls.append(match.start())
    return ind_ls
# 接受两个参数，ls 表示位置列表，start 表示起始位置。
# 遍历位置列表，筛选在ORF中的位置，使得位置与ORF起始位置的差值为3的倍数。
# 返回符合条件的位置列表 ind_inframe_ls。
def getInframeSubsls(ls,start):
    ind_inframe_ls = []
    for i in range(len(ls)):
        if abs(ls[i]-start) % 3 ==0:
            ind_inframe_ls.append(ls[i])
    return ind_inframe_ls
# 接受两个参数，seq_ls 表示包含多个序列的列表，ORF_loc_lsls 表示包含多个ORF位置信息列表的列表。
# 初始化类的属性，包括序列列表 seq_ls 和ORF位置信息列表的列表 ORF_loc_lsls。
class ORFs2GlobFea:

    def __init__(self,seq_ls,ORF_loc_lsls):

        self.seq_ls = seq_ls
        self.ORF_loc_lsls = ORF_loc_lsls
    # 遍历序列列表和ORF位置信息列表的列表，对于每个ORF位置信息，调用 orf2GlobFea 方法生成全局特征。
    # 将生成的全局特征存储在 glob_feas_lsls 列表中。
    # 返回最终的全局特征列表 glob_feas_lsls。
    def orfs2GloFea(self):
        glob_feas_lsls = []
        for i in range(len(self.seq_ls)):

            seq = self.seq_ls[i]
            ORF_loc_ls = self.ORF_loc_lsls[i]

            glob_feas_ls = []

            for j in range(len(ORF_loc_ls)):
                ORF_loc = ORF_loc_ls[j]

                glob_feas_ls.append(self.orf2GlobFea(seq,ORF_loc))

            glob_feas_lsls.append(glob_feas_ls)

        return glob_feas_lsls
    # 接受两个参数，seq 表示当前序列，orf_loc 表示当前ORF的位置信息。
    # 根据ORF的位置信息计算一系列全局特征，包括ORF起始位置、ORF结束位置、ORF长度等。
    # 利用 getSubstrLs 函数找到序列中的ATG起始位置，分别在ORF前半部分和后半部分计算ATG起始位置的数量。
    # 利用 getInframeSubsls 函数找到ATG起始位置中在ORF内的数量。
    # 使用类似的方式计算TAA、TAG、TGA的数量和在ORF内的数量。
    # 最后计算一些比率和长度相关的特征，并返回这些特征的数组。
    def orf2GlobFea(self,seq,orf_loc):
     # orf_loc: (ORF_start, ORF_end, integrity, label,ORF_frame)
     G0 = orf_loc[0]
     G1 = len(seq) - orf_loc[1]
     G2 = math.log((G1+1e-100)/(G0 + 1e-100))

     ind_ls0 = getSubstrLs('ATG',seq[0:G0])
     ind_ls1 = getSubstrLs('ATG',seq[G0:])

     G3 = len(ind_ls0)
     G4 = len(ind_ls1)

     G5 = math.log((G4+1e-100)/(G3+1e-100))

     ind_ls2 = getInframeSubsls(ind_ls0,G0)

     ind_ls3 = getInframeSubsls(ind_ls1,G0)

     G6 = len(ind_ls2)

     G7 = len(ind_ls3)

     G8 = math.log((G7+1e-100)/(G6+1e-100))

     ind_ls4 = getSubstrLs('TAA',seq[0:G0])
     ind_ls5 = getSubstrLs('TAG',seq[0:G0])
     ind_ls6 = getSubstrLs('TGA',seq[0:G0])

     G9 = len(ind_ls4) + len(ind_ls5) + len(ind_ls6)

     ind_ls7 = getSubstrLs('TAA',seq[G0:])
     ind_ls8 = getSubstrLs('TAG',seq[G0:])
     ind_ls9 = getSubstrLs('TGA',seq[G0:])

     G10 = len(ind_ls7) + len(ind_ls8) + len(ind_ls9)

     G11 = math.log((G10+1e-100)/(G9+1e-100))

     ind_ls10 = getInframeSubsls(ind_ls4, G0)
     ind_ls11 = getInframeSubsls(ind_ls5, G0)
     ind_ls12 = getInframeSubsls(ind_ls6, G0)

     G12 = len(ind_ls10) + len(ind_ls11) + len(ind_ls12)

     ind_ls13 = getInframeSubsls(ind_ls7, G0)
     ind_ls14 = getInframeSubsls(ind_ls8, G0)
     ind_ls15 = getInframeSubsls(ind_ls9, G0)

     G13 = len(ind_ls13) + len(ind_ls14) + len(ind_ls15)

     G14 = math.log((G13+1e-100)/(G12+1e-100))

     G15 = orf_loc[1] - orf_loc[0]

     G16 = G15/(len(seq))

     G17 = orf_loc[2]

     return np.array([G0,G1,G2,G3,G4,G5,G6,G7,G8,G9,G10,G11,G12,G13,G14,G15,G16,G17])


# f2o = fa2orfs.Fa2orfs('fasta/test.fa',True)
# seq_ls, lsls = f2o.seq2orfs()
# o2g = ORFs2GlobFea(seq_ls,lsls)
# glob_feas = o2g.orfs2GloFea()
#
# import numpy as np
# print(np.array(glob_feas))

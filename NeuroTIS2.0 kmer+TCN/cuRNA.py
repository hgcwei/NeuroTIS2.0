import numpy as np

class CuRNA:

    # 初始化 CuRNA 对象，接受两个参数：valid_letters 和 ws。
    # valid_letters 是一个包含RNA序列合法字母的列表，一般为 ['A', 'C', 'G', 'T']。
    # ws 表示滑动窗口的大小，一般为90，并要求其是18的倍数。
    def __init__(self,valid_letters,ws):

        self.k = 3
        self.valid_letters = valid_letters
        self.mul_by = len(valid_letters) ** np.arange(self.k-1, -1, -1)
        self.n = len(valid_letters)**self.k
        self.ws = ws     # 滑动窗大小，一般是90
        assert ws%18==0,"Sliding window size is multiple of 18!"

    # 输入参数为 rna（RNA序列）和 frameID（起始帧的编号），返回在指定帧中的ORF序列。
    # 确保RNA序列足够长，并根据指定的帧截取ORF。
    def get_orf(self,rna,frameID):
        assert len(rna) > 4, "Rna is too short!"
        assert frameID==0 or frameID==1 or frameID==2, "frameID is incorrect!"
        if len(rna[frameID:]) % 3 == 0:
            rna_exd = rna[frameID:]
        else:
            rna_exd = rna[frameID:] + 'N'*(3-(len(rna[frameID:]) % 3))
        return rna_exd
    # 输入参数为 le（ORF的长度）、cds（CDS的起始和终止位置）、frameID（ORF的起始帧编号）。
    # 返回一个二进制标签数组，表示在指定帧中CDS的位置
    def get_orf_lab(self,le,cds,frameID):
        assert len(cds)==2, 'length of cds is not 2!'
        label = np.zeros((le))
        s = [a for a in range(cds[0],cds[1],3)]
        label[s] = 1
        lab = label[[b for b in range(frameID,le,3)]]

        return lab
    # 输入参数为 cds（CDS的起始和终止位置），返回ORF的起始帧编号。
    def get_cds_frameID(self,cds):
        assert len(cds)==2, 'length of cds is not 2!'
        return cds[0]%3
    # 输入参数为 rna_exd（扩展后的RNA序列）。
    # 返回一个滑动窗口内每个位置的三联密码使用情况的数组。
    def get_slid_cu(self,rna_exd):

        # 在前面加一段，在后面加一段，防止越界
        rna_exd2 = 'N' * int(self.ws / 2) + rna_exd + 'N' * (int(self.ws / 2)-3)

        l = len(rna_exd2)
        # 一个六十四位的矩阵
        codon_usage = np.zeros((64))
        Q_inds = []
        # 遍历序列，步长为 3
        for i in range(0,self.ws,3):
            # print(i)
            ind = self.get_codon_ind(rna_exd2[i:(i+self.k)])
            # print(ind)
            Q_inds.append(ind)
            if ind > -1:
                codon_usage[ind]= codon_usage[ind] + 1

        codon_usage_ls = []
        codon_usage_ls.append(codon_usage.copy())

        for i in range(self.ws,l,3):
            ind_left = Q_inds[0]
            ind_right = self.get_codon_ind(rna_exd2[i:(i+self.k)])

            if ind_left>-1:
                codon_usage[ind_left] = codon_usage[ind_left] - 1
            if ind_right>-1:
                codon_usage[ind_right] = codon_usage[ind_right] + 1
            codon_usage_ls.append(codon_usage.copy())
            Q_inds.append(ind_right)
            del(Q_inds[0])

        cu_mat = np.array(codon_usage_ls)
        num = cu_mat.shape


        z = np.array([x / num[0] for x in range(num[0])])
        z = z[:,np.newaxis]
        zz = np.hstack((z,cu_mat))

        return zz
    # 输入参数为 kmer（三联密码），返回三联密码在 64 种可能密码中的索引。
    def get_codon_ind(self,kmer):
        digits = []
        for letter in kmer:
            if letter not in self.valid_letters:
                return -1
            digits.append(self.valid_letters.index(letter))

        digits = np.array(digits)
        pos = (digits*self.mul_by).sum()
        return pos

rna ='AAGGGTAGGACGCGGGGTA'

cur = CuRNA(['A','C','G','T'],18)
orf1 = cur.get_orf(rna,2)
print(orf1)
s = cur.get_slid_cu(orf1)
print(s.shape)
x = np.array([x / 6 for x in range(6)])

x = x[:,np.newaxis]
print(x.shape)

#
# s = 'AACGTTTAATCCG'
# cds = [3,8]
#
#
# cur = CuRNA(['A','C','G','T'],18)
# print(cur.get_orf_lab(len(s),cds,0))
# import random
# print(random.randint(0, int(1 / 0.5)))

# print([x/10 for x in range(10)])

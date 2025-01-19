import numpy as np
import fa2orfs
import pickle
import random, string

# 这段代码的目的是将输入的ORF数据按照指定条件转换为样本集，并且可以控制正负例的比例。

def random_string(length:int) -> str:
    """
    length: 指定随机字符串长度
    """
    random_str = ''
    # base_str = string.digits
    # base_str = string.ascii_letters
    base_str = string.digits + string.ascii_letters
    for i in range(length):
        random_str += random.choice(base_str)
    return random_str

def list2pickle(lst, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(lst, file)

class ORFs2Samples:

    # 构造函数 __init__ 接受参数 ORFs_loc_lsls（包含多个ORF的列表的列表）、type（指定ORF的类型）、
    # neg_pos_ratio（负例和正例的比例）
    def __init__(self, ORFs_loc_lsls, type , neg_pos_ratio=1.0):
        assert type == 0 or type == 1 or type==2 or type is None
        self.ORFs_loc_lsls = ORFs_loc_lsls
        self.neg_pos_ratio = neg_pos_ratio
        self.type = type

    # 遍历输入的 ORFs_loc_lsls，对每个ORF列表调用 orf2sample2 方法，将其转换为样本集。
    def orfs2samples(self):

        lsls = []
        for i in range(len(self.ORFs_loc_lsls)):

            lsls.append(self.orf2sample2(self.ORFs_loc_lsls[i]))

        return lsls
    #
    # def orf2sample(self,ls):
    #
    #     sl = []
    #     # ORF_start, ORF_end, integrity, type, label
    #     l = len(ls)
    #     ls2mat = np.array(ls)
    #     pos_index = np.where(ls2mat[:, 3] == 0)[0]
    #     if self.type==4:
    #         neg_index = np.where(ls2mat[:,3]>0)[0]
    #     else:
    #         neg_index = np.where(ls2mat[:,3]==self.type)[0]
    #     ls2 = list(neg_index)
    #     random.shuffle(ls2)
    #     ls3 = list(pos_index) + ls2
    #     ls4 = ls3[0:(int(self.neg_pos_ratio) + 1)]
    #     for i in range(len(ls4)):
    #         sl.append(ls[ls4[i]] + ['ORF_' +random_string(5)])
    #     return sl
    # 接受一个ORF列表 ls 作为输入，将其转换为样本集。
    # 创建一个空列表 sl 用于存储样本集。
    # 将输入的ORF列表 ls 转换为 NumPy 数组 ls2mat。
    # 根据条件筛选正例和负例的索引：
    # 正例条件：ls2mat[:, 4] == 1 且 ls2mat[:, 3] == 0。
    # 负例条件：ls2mat[:, 4] == 0 或者 ls2mat[:, 3] == self.type（根据 type 的不同）。
    # 将正例和负例索引取交集，并将其转换为列表 pos_index。
    # 对负例索引进行随机打乱，并转换为列表 ls2。
    # 将正例和负例的索引合并为列表 ls3。
    # 从合并后的索引列表中选取一定数量的索引，数量由 int(self.neg_pos_ratio) + len(pos_index) 决定。
    # 遍历选取的索引，将对应位置的ORF信息加入 sl 列表，并为每个样本生成一个唯一标识。
    # 返回最终的样本集 sl
    def orf2sample2(self,ls):
        sl = []
        ls2mat = np.array(ls)

        set0 = set(np.where(ls2mat[:,4]==1)[0])
        set1 = set(np.where(ls2mat[:,3]==0)[0])
        pos_index  = set0.intersection(set1)
        pos_index  = list(pos_index)
        if self.type is None:

            neg_index = np.where(ls2mat[:,4]==0)[0]

        else:
            set2 = set(np.where(ls2mat[:,4]==0)[0])
            set3 = set(np.where(ls2mat[:,3]==self.type)[0])
            neg_index = set2.intersection(set3)
        ls2 = list(neg_index)
        random.shuffle(ls2)
        ls3 = pos_index + ls2
        if self.neg_pos_ratio is None:
            ls4 = ls3
        else:
            ls4 = ls3[0:(int(self.neg_pos_ratio) + len(pos_index))]
        for i in range(len(ls4)):
            sl.append(ls[ls4[i]] + ['ORF_' +random_string(5)])
        return sl






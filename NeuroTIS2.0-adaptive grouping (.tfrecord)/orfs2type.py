
import numpy as np


def compute_frame_from_cs(cs0,cs1,cs2):
    # 这个函数接受三个参数 cs0、cs1、cs2，分别代表在三个不同的阅读框（frame）上的分数。
    # 它计算每个阅读框上分数的平均值，并将结果保存在 cs_mean 中。
    # 然后，根据平均分数的最大值，确定一个代表阅读框的数组 k，其中将具有最大平均分数的位置设置为1。
    # 最后，根据 k 的值，返回一个代表阅读框的数组，即 [0, 1, 2]、[2, 0, 1] 或 [1, 2, 0]。
    k = [0,0,0]
    cs_mean = (np.mean(cs0),np.mean(cs1),np.mean(cs2))
    k[np.where(cs_mean == np.max(cs_mean))[0][0]] = 1
    if k[0] == 1 and k[1] == 0 and k[2] == 0:
        return [0, 1, 2]

    if k[0] == 0 and k[1] == 1 and k[2] == 0:
        return [2, 0, 1]

    if k[0] == 0 and k[1] == 0 and k[2] == 1:
        return [1, 2, 0]


class ORFs2Type:

    # orfs_loc_lsls 是一个包含多个列表的列表，每个内部列表包含编码序列的位置信息。
    # cs_ls 是一个包含多个三元组的列表，每个三元组包含三个阅读框上的分数。
    def __init__(self,orfs_loc_lsls,cs_ls):

        # 分数
        self.cs_ls = cs_ls
        # 编译序列，位于启动子终止子之间
        self.orfs_loc_lsls = orfs_loc_lsls

    # 这个方法用于生成编码序列的类型列表。
    # 使用 for 循环迭代 self.orfs_loc_lsls 中的每个编码序列信息。
    # 对于每个编码序列，获取相应的三元组 cs0, cs1, cs2。
    # 调用 compute_frame_from_cs 函数获得阅读框的类型。
    # 对于每个编码序列中的每个位置信息，将其加入一个列表 ls，其中包含原始位置信息、阅读框类型和其他信息。
    # 将每个编码序列的 ls 列表加入到最终的结果列表 lsls 中。
    def gen_type_for_lsls(self):

        lsls = []

        for i in range(len(self.orfs_loc_lsls)):

            cs0, cs1, cs2 = self.cs_ls[i]
            # 012 / 120 / 210
            frame = compute_frame_from_cs(cs0,cs1,cs2)

            ls = []

            for j in range(len(self.orfs_loc_lsls[i])):

                orf_loc = self.orfs_loc_lsls[i][j]
                # tp = orf_loc[3]

                # if orf_loc[4]==0:
                tp = frame[orf_loc[0]%3]

                ls.append([orf_loc[0],orf_loc[1],orf_loc[2],tp,orf_loc[4]])

                # if orf_loc[4]== 1:

                    # if frame[orf_loc[0]%3] + orf_loc[4] != 1:
                    #     print(frame[orf_loc[0]%3],orf_loc[4])

            lsls.append(ls)

        return lsls

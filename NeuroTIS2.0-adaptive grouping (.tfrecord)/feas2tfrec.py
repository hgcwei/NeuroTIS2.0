import tensorflow as tf
# 返回一个tf.train.Feature对象，用于处理整数特征。
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# 返回一个tf.train.Feature对象，用于处理字符串特征。
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

class Feas2TFRecord:
    # 接受五个参数，分别是c4_feas_lsls（c4 特征的列表的列表）、
    # cod_feas_lsls（编码特征的列表的列表）、glob_feas_lsls（全局特征的列表的列表）、
    # ORFs_loc_lsls（ORF位置信息的列表的列表）、tfrec_name（TFRecord文件的名称）
    def __init__(self,c4_feas_lsls, cod_feas_lsls, glob_feas_lsls, ORFs_loc_lsls, tfrec_name):
        import numpy as np
        self.c4_feas_lsls = c4_feas_lsls
        self.cod_feas_lsls = cod_feas_lsls
        self.glob_feas_lsls = glob_feas_lsls
        self.ORFs_loc_lsls = ORFs_loc_lsls
        self.tfrec_name = 'tfrecs/' + tfrec_name
    # 创建一个TFRecord写入器 writer，使用 tf.compat.v1.python_io.TFRecordWriter 类。
    # 遍历c4、编码、全局特征以及ORF位置信息的列表的列表，以及它们的子列表。
    # 对于每个子列表，遍历其中的元素，并将每个元素的特征转换为一个 tf.train.Example 对象。
    # 使用 _bytes_feature 处理字节字符串特征，使用 _int64_feature 处理整数特征。
    # 将所有特征组成的 tf.train.Example 对象序列化为字符串，并写入TFRecord文件。
    # 关闭TFRecord写入器。
    def feas_lsls2tfrecs(self):

        writer = tf.compat.v1.python_io.TFRecordWriter(self.tfrec_name)
        no = 0
        for i in range(len(self.c4_feas_lsls)):

            c4_feas_ls = self.c4_feas_lsls[i]
            cod_feas_ls = self.cod_feas_lsls[i]
            glob_feas_ls = self.glob_feas_lsls[i]
            ORFs_loc_ls = self.ORFs_loc_lsls[i]

            for j in range(len(c4_feas_ls)):

                c4_feas = c4_feas_ls[j]
                cod_feas = cod_feas_ls[j]
                glob_feas = glob_feas_ls[j]
                ORFs_loc = ORFs_loc_ls[j]

                c4_ = c4_feas.tostring()
                cod_ = cod_feas.tostring()
                glo_ = glob_feas.tostring()
                lab = ORFs_loc[4]
                example = tf.train.Example(features=tf.train.Features(feature={
                    'c4_': _bytes_feature(c4_),
                    'cod_': _bytes_feature(cod_),
                    'glo_': _bytes_feature(glo_),
                    'lab': _int64_feature(lab),
                    'no_': _int64_feature(no)
                }))
                no = no + 1
                writer.write(example.SerializeToString())
        writer.close()




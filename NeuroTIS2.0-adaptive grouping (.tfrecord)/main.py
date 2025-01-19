import fa2orfs
import feas2tfrec
import orfs2c4TIS
import orfs2CodFea
import orfs2GlobFea
import orfs2samples
import rnas2cod
import orfs2type
import pandas as pd


def gen_tfrecord_for_fa(fname,tfrname,wds_c4=100,wds_cod=200,type=2,neg_pos_ratio=1.0):
    print(neg_pos_ratio)
    # wds_c4 = 100
    # wds_cod = 200
    # type = 2
    # neg_pos_ratio = None
    # fname = 'transcript-wide-mouse.fa.1'
    # tfrname = 'test.tp'+ str(type) +'.all.tfrec'
    # 使用fa2orfs模块中的Fa2orfs类从FASTA文件中读取原始序列，
    # 并找到其中的ORFs（开放阅读框）。
    f2o = fa2orfs.Fa2orfs('fasta/' + fname,True)
    # 返回的原序列，及启动子终止子之间的序列
    seq_ls, orfs_loc_lsls = f2o.seq2orfs()
    # 使用标准模型计算得分
    # 使用rnas2cod模块中的RNAs2cod类，利用预训练的模型计算RNA的编码得分。
    r2c = rnas2cod.RNAs2cod(seq_ls,'model/human.tcn.fea.64.fil.15.ker.20.dil.124.nb.1.len.2000.weights.0.9922471')
    # 得到的是分别的三个阅读框的得分
    cs_ls = r2c.gen_coding_scores_for_rnas()
    # 使用orfs2type模块中的ORFs2Type类，结合ORF位置和RNA编码得分，确定每个ORF的类型。
    o2y = orfs2type.ORFs2Type(orfs_loc_lsls,cs_ls)
    # 其中包含原始位置信息、阅读框类型
    lsls = o2y.gen_type_for_lsls()

    # 使用orfs2samples模块中的ORFs2Samples类，基于确定的ORF类型，生成样本集。
    o2s = orfs2samples.ORFs2Samples(lsls,type,neg_pos_ratio)
    lsls2 = o2s.orfs2samples()

    # 使用orfs2c4TIS模块中的ORFs2C4TIS类，生成C4特征。
    o2t = orfs2c4TIS.ORFs2C4TIS(seq_ls,lsls2,wds_c4)
    c4_feas_lsls = o2t.gen_c4_for_tiss()

    # 使用orfs2CodFea模块中的ORFs2CodFea类，基于编码得分，生成相应的特征。
    o2c = orfs2CodFea.ORFs2CodFea(seq_ls,lsls2,wds_cod,cs_ls)
    cod_feas_lsls = o2c.gen_cs_for_tis_pred()

    # 使用orfs2GlobFea模块中的ORFs2GlobFea类，生成全局特征。
    o2g = orfs2GlobFea.ORFs2GlobFea(seq_ls,lsls2)
    glo_feas_lsls = o2g.orfs2GloFea()

    # 使用feas2tfrec模块中的Feas2TFRecord类，
    # 将生成的C4、编码、全局特征以及相应的标签转换为TFRecord格式，以便于TensorFlow的处理。
    f2t = feas2tfrec.Feas2TFRecord(c4_feas_lsls,cod_feas_lsls,glo_feas_lsls,lsls2,tfrname)
    f2t.feas_lsls2tfrecs()

# t0.tp.None.np.1.0.tfrec 中，t0表示文件编号，tp None表示 类型是随机的（012都有可能）， np 1.0 表示正负样本比例为1.0

gen_tfrecord_for_fa('human/transcript.human.nonrec.fa.4','t4.tp.2.np.None.NeuroTIS_plus.human.tfrec',neg_pos_ratio=None, wds_c4=100, type=2)
# gen_tfrecord_for_fa('transcript-wide-mouse.fa.0.1','t1.tp.None.np.1.0.tfrec',wds_c4=400,type=None)
# gen_tfrecord_for_fa('transcript-wide-mouse.fa.0.2','t2.tp.None.np.1.0.tfrec',wds_c4=400,type=None)
# gen_tfrecord_for_fa('transcript-wide-mouse.fa.0.3','t3.tp.None.np.1.0.tfrec',wds_c4=400,type=None)
# gen_tfrecord_for_fa('transcript-wide-mouse.fa.1','t4.tp.None.np.1.0.tfrec',wds_c4=400,type=None)
#
#
# gen_tfrecord_for_fa('transcript-wide-mouse.fa.1','t4.tp.None.np.1.0.all.tfrec',wds_c4=400,type=None,neg_pos_ratio=None)
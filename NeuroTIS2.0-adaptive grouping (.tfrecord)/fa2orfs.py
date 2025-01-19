# **************************************************************************************************************************
#  Author: Chao Wei
#  Function: this code extracts all the possible integrative ORFs of a fasta file
#  ORF（开放阅读框） 是理论上的氨基酸编码区，一般是在分析基因序列得到的。把基因的mRNA输入到程序种，
#  程序会自动在 DNA 序列中寻找启动子（ATG 或 AUG），然后按每 3 个核酸一组，一直延伸寻找下去，
#  直到碰到终止子（TAA 或 TAG）。此时程序就把这个区域当成 一个ORF 区，认为理论上可以编码一组氨基酸。
#  Input:   fasta file
#  Output:  list [(start1, end1),(start2, end2),...]
#  paper:  NeruoTIS+: ...
# **************************************************************************************************************************

# 该代码模块实现了从FASTA文件中提取综合ORFs的功能。通过使用BioPython库中的SeqIO模块读取FASTA文件，
# 然后根据提供的启动子和终止子信息，提取序列中的ORFs。
# 在提取的同时，如果启动子和终止子处于CDS范围内，将标签设置为1。
# 最后，返回提取的ORF位置列表和对应的序列列表。

from Bio import SeqIO
import re

# 定义一个函数，从FASTA文件中获取CDS（Coding DNA Sequence）的起始和终止位置
def getCDS(header):
    cds = re.findall('\d+', header)
    return (int(cds[0])-1,int(cds[1]))

class Fa2orfs:
    # 初始化函数，接收FASTA文件路径和是否使用CDS的标志
    def __init__(self,fa,hs_cds):
        self.fa = fa
        self.hs_cds = hs_cds

    # 生成每个三联密码子的位置
    def codons(self, seq, frame):
        start_coord = frame
        codons_ls = []
        while start_coord + 3 <= len(seq):
            # yield (self.seq[start_coord:start_coord + 3], start_coord)
            codons_ls.append((seq[start_coord:start_coord + 3], start_coord))
            start_coord += 3
        return codons_ls

    # 将整个序列文件中的ORFs提取出来
    def seq2orfs(self,start_codon=['ATG'],stop_codon=['TAA', 'TAG', 'TGA']):
        orf_loc_lsls = []
        seq_ls = []
        for rec in SeqIO.parse(self.fa, 'fasta'):
            seq = str.upper(str(rec.seq))
            orf_loc_ls0 = self.seq2orf(seq, 0, start_codon, stop_codon)
            orf_loc_ls1 = self.seq2orf(seq, 1, start_codon, stop_codon)
            orf_loc_ls2 = self.seq2orf(seq, 2, start_codon, stop_codon)

            if self.hs_cds:
                cds = getCDS(rec.description)
                orf_loc_ls0 = self.set_label_type(orf_loc_ls0,cds)
                orf_loc_ls1 = self.set_label_type(orf_loc_ls1,cds)
                orf_loc_ls2 = self.set_label_type(orf_loc_ls2,cds)

            orf_loc_ls = orf_loc_ls0 + orf_loc_ls1 + orf_loc_ls2

            orf_loc_lsls.append(orf_loc_ls)
            seq_ls.append(seq)

        return seq_ls, orf_loc_lsls

    # 根据给定的起始密码子和终止密码子提取ORF
    def seq2orf(self, seq, frame_number, start_codon, stop_codon):
        codon_posi = self.codons(seq,frame_number)
        start_codons = start_codon
        stop_codons = stop_codon
        orf_loc_ls = []
        type = 0
        label = 0
        i = -1
        while True:

            try:
                # codon, index = codon_posi.__next__()
                i = i+1
                j = i
                codon, index = codon_posi[i]

            except:
                break
            if codon in start_codons and codon not in stop_codons:
                ORF_start = index
                end = False
                while True:
                    try:

                        # codon, index = codon_posi.__next__()
                        j = j + 1
                        codon,index = codon_posi[j]

                    except:
                        end = True
                        integrity = -1
                    if codon in stop_codons:
                        integrity = 1
                        end = True
                    if end:
                        ORF_end = index + 3
                        orf_loc_ls.append([ORF_start, ORF_end, integrity ,type,label])
                        # if integrity == -1:
                        #     print('exist unintegrative!')
                        break

        return orf_loc_ls

    # 对ORFs进行标签设置，以及CDS范围内的设置
    def set_label_type(self, orf_loc_ls, seq_cds):
        for i in range(len(orf_loc_ls)):
            start = orf_loc_ls[i][0]
            end = orf_loc_ls[i][1]
            if start == seq_cds[0] and end == seq_cds[1]:
                #orf_loc_ls[i][3] = 0
                orf_loc_ls[i][4] = 1
            # if start < seq_cds[0] or start > seq_cds[1]:
            #     orf_loc_ls[i][3] = 4

            # if start > seq_cds[0] and start < seq_cds[1]:
            #     orf_f = (start - seq_cds[0]) % 3
                # orf_loc_ls[i][3] = orf_f + 1
        return orf_loc_ls



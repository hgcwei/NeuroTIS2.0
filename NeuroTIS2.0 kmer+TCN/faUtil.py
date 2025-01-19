from Bio import SeqIO
import cuRNA
import random
import numpy as np

def cut_fasta(fa,ratio):
    assert isinstance(fa,str),'test 001, variable fa should be string!'
    assert ratio > 0 and ratio < 1, 'test 002, ratio is betweet 0 and 1!'
    recs = []
    for rec in SeqIO.parse(fa, 'fasta'):
        recs.append(rec)
    assert len(recs) > 0, 'test 003, fasta should be not empty!'
    num_c = int(len(recs) * ratio)

    SeqIO.write(recs[:num_c], fa + '.ratio.'+str(round(ratio,1))+'.00', 'fasta')
    SeqIO.write(recs[num_c:], fa + '.ratio.'+str(round(1-ratio,1))+'.01', 'fasta')

def get_cds_from_str(header):
    assert header[0]=='[','test 004, header format is wrong!'
    s = header[header.find('[')+1: header.find(']')]
    s = s.split(" ")
    return [int(s[0])-1,int(s[-1])-1]

def cut_fasta2(fa,ratio_ls):

    assert isinstance(fa,str),'test 001, variable fa should be string!'
    assert sum(ratio_ls)<=1 and sum(ratio_ls)>=0.999, 'test 002, sum of ratio if not 1!'

    recs = []
    for rec in SeqIO.parse(fa, 'fasta'):
        recs.append(rec)
    num = len(recs)
    assert num > 0, 'test 003, fasta should be not empty!'

    num_ls = list(map(int,(np.floor(np.array(ratio_ls[:-1])*num)).tolist()))

    num_ls_new = num_ls + [num - sum(num_ls)]

    for i in range(len(num_ls_new)):
        n = num_ls_new[i]
        recs_tmp = recs[0:n]
        SeqIO.write(recs_tmp, fa + '.' + str(i), 'fasta')
        del recs[0:n]


class FaUtil:

    def __init__(self,fa,sample_type,sample_frq=1/6):
        assert sample_type=='bal' or sample_type=='nbal', 'class FaUtil test 001: type has problem!'
        assert sample_frq>0 and sample_frq < 1, 'class FaUtil test 002: sample frequenct should be 0 to 1!'
        self.fa = fa
        self.cu = cuRNA.CuRNA(['A','C','G','T'],90)
        self.type = sample_type
        self.frq = sample_frq


    def gen_codon_usage_lab_fa(self):

        mat_cus = []
        lab_cs = []
        wid_s = []

        for rec in SeqIO.parse(self.fa,'fasta'):
            rna = rec.seq

            cds = get_cds_from_str(rec.description)

            if self.type == 'bal':

                correct_frameID = self.cu.get_cds_frameID(cds)
                correct_orf = self.cu.get_orf(rna,correct_frameID)
                self.cu.get_orf_lab(len(rna),cds,correct_frameID)
                mat_cu_c = self.cu.get_slid_cu(correct_orf)
                lab_c = self.cu.get_orf_lab(len(rna),cds,correct_frameID)

                if (random.randint(0, int(1 / self.frq - 1)) == 0):

                    ids = [0,1,2]
                    ids.remove(correct_frameID)
                    frameID_nc = ids[random.randint(0,1)]

                    orf_nc = self.cu.get_orf(rna,frameID_nc)
                    mat_cu_nc = self.cu.get_slid_cu(orf_nc)
                    lab_nc = self.cu.get_orf_lab(len(rna),cds,frameID_nc)

                    mat_cus.append(np.vstack((mat_cu_c,mat_cu_nc)))
                    lab_cs.append(np.hstack((lab_c.reshape(1,-1),lab_nc.reshape(1,-1))))
                else:
                    mat_cus.append(mat_cu_c)
                    lab_cs.append(lab_c.reshape(1,-1))

            else:

                orf0 = self.cu.get_orf(rna, 0)
                orf1 = self.cu.get_orf(rna, 1)
                orf2 = self.cu.get_orf(rna, 2)

                mat_cu0 = self.cu.get_slid_cu(orf0)
                mat_cu1 = self.cu.get_slid_cu(orf1)
                mat_cu2 = self.cu.get_slid_cu(orf2)

                lab_c0 = self.cu.get_orf_lab(len(rna),cds,0)
                lab_c1 = self.cu.get_orf_lab(len(rna),cds,1)
                lab_c2 = self.cu.get_orf_lab(len(rna),cds,2)

                wid_s.append((len(lab_c0),len(lab_c1),len(lab_c2)))
                mat_cus.append(np.vstack((mat_cu0,mat_cu1,mat_cu2)))
                lab_cs.append(np.hstack((lab_c0.reshape(1,-1),lab_c1.reshape(1,-1),lab_c2.reshape(1,-1))))

        return np.vstack((mat_cus)),np.hstack(lab_cs),wid_s


    def gen_codon_usage_for_test(self):

        mat_cus = []
        wid_s = []

        for rec in SeqIO.parse(self.fa,'fasta'):
            rna = rec.seq


            orf0 = self.cu.get_orf(rna, 0)
            orf1 = self.cu.get_orf(rna, 1)
            orf2 = self.cu.get_orf(rna, 2)

            mat_cu0 = self.cu.get_slid_cu(orf0)
            mat_cu1 = self.cu.get_slid_cu(orf1)
            mat_cu2 = self.cu.get_slid_cu(orf2)

            # kmer得分序列，拼接 行数拼接？？？
            wid_s.append((mat_cu0.shape[0], mat_cu1.shape[0], mat_cu2.shape[0]))
            mat_cus.append(np.vstack((mat_cu0, mat_cu1, mat_cu2)))

        return np.vstack((mat_cus)),wid_s

#
#
fu0 = FaUtil('fasta/human/transcript.human.nonrec.fa.4','bal',1/6)
mat0,lab0,_ = fu0.gen_codon_usage_lab_fa()
#
# fu1 = FaUtil('fasta/human/transcript.human.nonrec.fa.1','bal',1/6)
# mat1,lab1,_ = fu1.gen_codon_usage_lab_fa()


# np.savetxt('human.train_kmer64.csv',np.transpose(np.vstack((mat0,mat1))),fmt='%.4f',delimiter=',')
# np.savetxt('human.train_label64.csv',np.hstack((lab0,lab1)),fmt='%d',delimiter=',')

#
np.savetxt('human.test_kmer64.csv',np.transpose(mat0),fmt='%.4f',delimiter=',')
np.savetxt('human.test_label64.csv',lab0,fmt='%d',delimiter=',')


# print(get_cds_from_str('[5 10]c.wei'))

# cut_fasta2('fasta/Human.ncrna.train.fa',[0.4,0.3,0.3])
# s = [0.1,0.2]
#
# import numpy as np
#
# print(np.mean(np.array(s)))
# print( isinstance(s,list))


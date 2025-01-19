from Bio import SeqIO
import numpy as np

def cut_fasta(fa,ratio_ls):

    assert isinstance(fa,str),'test 001, variable fa should be string!'
    assert sum(ratio_ls)<=1 and sum(ratio_ls)>=0.999, 'test 002, sum of ratio if not 1!'

    recs = []
    for rec in SeqIO.parse(fa, 'fasta'):
        recs.append(rec)
    num = len(recs)
    print(num)
    assert num > 0, 'test 003, fasta should be not empty!'

    num_ls = list(map(int,(np.floor(np.array(ratio_ls[:-1])*num)).tolist()))

    num_ls_new = num_ls + [num - sum(num_ls)]

    # for i in range(len(num_ls_new)):
    #     n = num_ls_new[i]
    #     recs_tmp = recs[0:n]
    #     SeqIO.write(recs_tmp, fa + '.' + str(i), 'fasta')
    #     del recs[0:n]


cut_fasta('transcript-mouse-cd-hit -i transcript-wide-mouse.fa -o new.fa -c 0.8.fa',[0.5,0.5])
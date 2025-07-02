# NeuroTIS+

A improved version of NeuroTIS, which is developed for translation initiation site prediction in full-length mRNA sequence.


corresponding author: Chao Wei
e-mail:weichao.2022@hbut.edu.cn

kmer+TCN is implemented based on https://github.com/philipperemy/keras-tcn, Thanks!

1. Setup
python 3.6.13
numpy 1.19.2
keras 2.3.0
keras 3.5.0
          

3. .fasta input
    please input a fa file.


5. Feature Generation
   Generate tfrecord (features) by running main.py in 

6. TIS Prediction

   Predict TIS by running tisTest.py in folder NeuroTIS2.0 frame-specific CNN

import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


wds_c4 = 100
wds_cod = 200

def parse_tfrecord(filename_ls):
    filename_queue = tf.train.string_input_producer(filename_ls, shuffle=True)  # 生成一个queue队列

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
            serialized_example,
            features={
          'c4_': tf.FixedLenFeature([], tf.string),
          'cod_': tf.FixedLenFeature([], tf.string),
          'glo_': tf.FixedLenFeature([], tf.string),
          'lab': tf.FixedLenFeature([], tf.int64),
          'no_': tf.FixedLenFeature([],tf.int64)
                    }
            )
    c4_fea = tf.reshape(tf.decode_raw(features['c4_'],tf.uint8),[wds_c4,4,1])
    cod_fea = tf.reshape(tf.decode_raw(features['cod_'],tf.float64),[400,])
    glo_fea = tf.reshape(tf.decode_raw(features['glo_'],tf.float64),[18,])

    return  c4_fea, cod_fea,  glo_fea, features['lab'],features['no_']

def count_tfrecord_number(tf_records_ls):
    c = 0
    for fn in tf_records_ls:
        for record in tf.python_io.tf_record_iterator(fn):
            c += 1
    return c


def network_neurotis_plus(x1, x2, x3):
    conv31 = tf.layers.conv2d(inputs=x1, filters=50, kernel_size=[5, 4], padding="valid", activation=tf.nn.relu,
                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool31 = tf.layers.max_pooling2d(inputs=conv31, pool_size=[3, 1], strides=3)
    drop31 = tf.layers.dropout(pool31, 0.1)

    conv32 = tf.layers.conv2d(inputs=drop31, filters=50, kernel_size=[3, 1], padding="valid",
                              activation=tf.nn.relu,
                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool32 = tf.layers.max_pooling2d(inputs=conv32, pool_size=[3, 1], strides=3)
    drop32 = tf.layers.dropout(pool32, 0.1)

    conv33 = tf.layers.conv2d(inputs=drop32, filters=50, kernel_size=[5, 1], padding="valid",
                              activation=tf.nn.relu,
                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool33 = tf.layers.max_pooling2d(inputs=conv33, pool_size=[3, 1], strides=3)
    drop33 = tf.layers.dropout(pool33, 0.1)

    re31 = tf.reshape(drop33, [-1, 2 * 50])

    zzz0 = tf.reshape(x2,[-1,400,1,1])

    zzz1 = tf.layers.conv2d(inputs=zzz0, filters=50, kernel_size=[7,1], padding="valid", activation=tf.nn.relu,
                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

    zzz2 = tf.reshape(zzz1, [-1, 394 * 50])




    co1 = tf.concat([re31, zzz2, x3], axis=1)

    dense31 = tf.layers.dense(inputs=co1, units=200, activation=tf.nn.relu,
                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

    drop34 = tf.layers.dropout(dense31, 0.1)
    logits = tf.layers.dense(inputs=drop34, units=2, activation=None,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    _y = tf.nn.softmax(logits)

    return _y, logits


h1 = 100
w1 = 4
c = 1

h2 = 400
w2 = 1

h3 = 18
w3 = 1

batch_size = 500
n_epoch = 15


type = '0'

train_tfrecords_ls = ['tfrecs/human/'+type+'/t0.tp.' +type + '.np.1.0.tfrec','tfrecs/human/'+type+'/t1.tp.' +type + '.np.1.0.tfrec','tfrecs/human/'+type+'/t2.tp.' +type + '.np.1.0.tfrec','tfrecs/human/'+type+'/t3.tp.' +type + '.np.1.0.tfrec']

test_tfrecords_ls = ['tfrecs/human/'+type+'/t4.tp.' +type + '.np.1.0.tfrec']


x1 = tf.placeholder(tf.float32, shape=[None, h1, w1, c], name='x1')
x2 = tf.placeholder(tf.float32, shape=[None, h2], name='x2')
x3 = tf.placeholder(tf.float32,shape=[None, h3], name = 'x3')
y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')


# x21, x22 = tf.split(x2,[200,200],axis=1)
x2_resh = tf.reshape(x2, [-1, h2])
x3_resh = tf.reshape(x3, [-1, h3])

y_hat, logits = network_neurotis_plus(x1,x2_resh,x3_resh)
loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=tf.one_hot(y_,2),logits=logits,pos_weight=1.0))

train_op = tf.train.AdamOptimizer(learning_rate=5e-4).minimize(loss)
correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

c4_,cod_, glo_,lab_, no_ = parse_tfrecord(train_tfrecords_ls)

c4_bat,cod_bat,  glo_bat,lab_bat, no_bat = tf.train.shuffle_batch([c4_, cod_,   glo_, lab_, no_], batch_size=batch_size, capacity=5000, min_after_dequeue=1000)

# c2_test, kmer_test, label_test = parse_tfrecord('C:/Users/Wei/PycharmProjects/tfrecord_test/TestFinal/test.tfrecords')
c4_tst,cod_tst, glo_tst,lab_tst, no_tst = parse_tfrecord(test_tfrecords_ls)
c4_bat_tst,cod_bat_tst, glo_bat_tst,lab_bat_tst,no_bat_tst = tf.train.shuffle_batch([c4_tst, cod_tst, glo_tst,lab_tst,no_tst],
                                                                          batch_size=batch_size, capacity=50000,
                                                                          min_after_dequeue=1000)

# 训练和测试数据，可将n_epoch设置更大一些
sess = tf.InteractiveSession()

saver = tf.train.Saver()
import time
start = time.clock()

total_num = count_tfrecord_number(train_tfrecords_ls)
with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    batch_idxs = int(total_num / batch_size)
    # batch_idxs = int(2533951 / batch_size)

    for epoch in range(n_epoch):
        train_loss, train_acc, train_batch = 0, 0, 0
        val_loss, val_acc, val_batch = 0, 0, 0
        for j in range(batch_idxs):
            c4_bats, cod_bats, glo_bats, lab_bats, no_bats = sess.run([c4_bat, cod_bat,glo_bat,lab_bat,no_bat])
            _, err, ac = sess.run([train_op, loss, acc],
                                  feed_dict={x1: c4_bats, x2: cod_bats,  x3: glo_bats, y_: lab_bats})
            train_loss += err
            train_acc += ac
            train_batch += 1

            c4_bats_tst,cod_bats_tst,  glo_bats_tst, lab_bats_tst, no_bats_tst = sess.run(
                [c4_bat_tst,cod_bat_tst, glo_bat_tst,lab_bat_tst, no_bat_tst])


            err, ac = sess.run([loss, acc],
                               feed_dict={x1: c4_bats_tst, x2: cod_bats_tst,  x3: glo_bats_tst, y_: lab_bats_tst})
            val_loss += err
            val_acc += ac
            val_batch += 1

            if np.mod(j, 10) == 0:
                # if n_epoch==(epoch+1):
                    print("(%d/%d) train loss: %f, train acc: %f, validation loss: %f ,validation acc: %f" % (
                    n_epoch, epoch + 1, train_loss / train_batch, train_acc / train_batch, val_loss / val_batch,
                    val_acc / val_batch))

    end = time.clock()
    print("time elaspe: %s" % (end - start))
    saver.save(sess, './model/'+type+'/neurotis.tis.model')
    coord.request_stop()
    coord.join(threads)
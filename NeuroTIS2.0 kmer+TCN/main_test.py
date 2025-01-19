from uuid import uuid4
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import load_model
from tensorflow.keras.models import save_model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tcn import compiled_tcn,TCN
from utils import data_generator
from utils import data_gen_final
from tensorflow.keras.layers import RepeatVector


# x_train, y_train = data_generator(601, 10, 30000)
# x_test, y_test = data_generator(601, 10, 6000)

# x_train,y_train,x_test,y_test = data_gen3('train_kmer_data.csv','train_kmer_labels.csv','test_kmer_data.csv','test_kmer_labels.csv')
# x_test,y_test = data_gen4('test_kmer_data.csv','test_kmer_labels.csv')

# x_train,y_train = data_gen_final('human.train_kmer64.csv','human.train_label64.csv',2000)

x_test,y_test = data_gen_final('test_kmer64.csv','test_label64.csv',2000)


# print(x_train.shape)
# print(x_train[0:1].shape[1])
print(x_test.shape)
# print(y_train.shape)

class PrintSomeValues(Callback):

    def on_epoch_begin(self, epoch, logs={}):
        print('y_true')
        print(np.array(y_test[:5, -10:].squeeze(), dtype=int))
        print('y_pred')
        print(self.model.predict(x_test[:5])[:, -10:].argmax(axis=-1))


def run_task():
    model = compiled_tcn(num_feat=65,
                         num_classes=2,
                         nb_filters=15,
                         kernel_size=20,
                         dilations=[1,2,4],
                         nb_stacks=2,
                         max_len=2000,
                         use_skip_connections=False,
                         opt='rmsprop',
                         padding='same',
                         lr=1e-3,
                         use_weight_norm=True,
                         return_sequences=True)
    # model = Sequential([
    #     TCN(input_shape=(5000,64),nb_filters=10,kernel_size=15,dilations=[1,2,4],nb_stacks=1,use_skip_connections=True,padding='same',use_weight_norm=True,return_sequences=True),
    #     # RepeatVector(5000),  # output.shape = (batch, output_timesteps, 64)
    #     Dense(2)
    # ])

    # print(f'x_train.shape = {x_train.shape}')
    # print(f'y_train.shape = {y_train.shape}')

    psv = PrintSomeValues()

    # Using sparse softmax.
    # http://chappers.github.io/web%20micro%20log/2017/01/26/quick-models-in-keras/
    # model.summary()

    # model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    # model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50,
    #           callbacks=[psv], batch_size=2)
    #
    # # save_model(model,'tcn.fea.64.fil.15.ker.15.dil.124.nb.1.keras')
    #

    #
    model.load_weights('model/tcn.fea.64.fil.10.ker.20.dil.124.nb.1.len.1000.weights.0.9962864')

    # model.summary()


    scores = model.predict(x=x_test)
    scores = scores.reshape([-1,2])
    yy_test = y_test.reshape([-1,1])
    print(scores.shape)
    np.savetxt('scores_kmer65_tcn.csv',scores,delimiter=',')
    np.savetxt('label_kmer65_tcn.csv',yy_test,delimiter=',')
    test_acc = model.evaluate(x=x_test, y=y_test)[1]  # accuracy.
    with open(f'coding_regions_{str(uuid4())[0:5]}.txt', 'w') as w:
        w.write(str(test_acc) + '\n')

    # model.save_weights('human.tcn.fea.64.fil.15.ker.20.dil.124.nb.1.len.2000.weights.'+ str(test_acc))


if __name__ == '__main__':
    run_task()

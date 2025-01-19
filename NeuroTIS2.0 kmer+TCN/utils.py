import numpy as np


def data_generator(t, mem_length, b_size):
    """
    Generate data for the copying memory task
    :param t: The total blank time length
    :param mem_length: The length of the memory to be recalled
    :param b_size: The batch size
    :return: Input and target data tensor
    """
    seq = np.array(np.random.randint(1, 9, size=(b_size, mem_length)), dtype=float)
    zeros = np.zeros((b_size, t))
    marker = 9 * np.ones((b_size, mem_length + 1))
    placeholders = np.zeros((b_size, mem_length))

    x = np.array(np.concatenate((seq, zeros[:, :-1], marker), 1), dtype=int)
    y = np.array(np.concatenate((placeholders, zeros, seq), 1), dtype=int)
    return np.expand_dims(x, axis=2).astype(np.float32), np.expand_dims(y, axis=2).astype(np.float32)

def data_gen(train_data_file,train_label_file,test_data_file,test_label_file):
    train_data = np.loadtxt(open(train_data_file, "rb"), delimiter=",", skiprows=0, dtype=np.float32)
    print(train_data.shape)
    train_data = train_data.transpose()
    train_data = train_data.reshape((-1,500,1))

    train_label = np.loadtxt(open(train_label_file, "rb"), delimiter=",", skiprows=0, dtype=np.int32)
    train_label = train_label.reshape((-1,500))

    test_data = np.loadtxt(open(test_data_file, "rb"), delimiter=",", skiprows=0, dtype=np.float32)
    test_data = test_data.transpose()
    test_data = test_data.reshape((-1,500,1))

    test_label = np.loadtxt(open(test_label_file, "rb"), delimiter=",", skiprows=0, dtype=np.int32)
    test_label = test_label.reshape((-1,500))

    return train_data, train_label, test_data, test_label

def data_gen3(train_data_file,train_label_file,test_data_file,test_label_file):
    train_data = np.loadtxt(open(train_data_file, "rb"), delimiter=",", skiprows=0, dtype=np.float32)
    print(train_data.shape)
    train_data = train_data.transpose()
    train_data = train_data.reshape((1000,8000,64))

    train_label = np.loadtxt(open(train_label_file, "rb"), delimiter=",", skiprows=0, dtype=np.int32)
    train_label = train_label.reshape((1000,8000))

    test_data = np.loadtxt(open(test_data_file, "rb"), delimiter=",", skiprows=0, dtype=np.float32)
    test_data = test_data.transpose()
    test_data = test_data.reshape((375,8000,64))

    test_label = np.loadtxt(open(test_label_file, "rb"), delimiter=",", skiprows=0, dtype=np.int32)
    test_label = test_label.reshape((375,8000))

    return train_data, train_label, test_data, test_label

def data_gen4(test_data_file,test_label_file):

    test_data = np.loadtxt(open(test_data_file, "rb"), delimiter=",", skiprows=0, dtype=np.float32)
    test_data = test_data.transpose()
    test_data = test_data.reshape((375,8000,64))

    test_label = np.loadtxt(open(test_label_file, "rb"), delimiter=",", skiprows=0, dtype=np.int32)
    test_label = test_label.reshape((375,8000))

    return  test_data, test_label
#
# if __name__ == '__main__':
#     print(data_generator(t=601, mem_length=10, b_size=1)[0].flatten())��

def data_gen2(train_data_file,train_label_file,test_data_file,test_label_file,longth):
    import math

    train_data = np.loadtxt(train_data_file, delimiter=' ',dtype=np.float32)
    train_data = np.reshape(train_data,[-1,1])
    train_label = np.loadtxt(train_label_file, delimiter=',',dtype=np.int32)
    train_label = np.reshape(train_label,[-1,1])

    num = math.floor(train_label.shape[0] / longth) * longth

    train_data = train_data[0:num,:]
    train_label = train_label[0:num,:]

    train_data = train_data.reshape((-1,longth,1))
    train_label = train_label.reshape((-1,longth))


    test_data = np.loadtxt(test_data_file, delimiter=' ',dtype=np.float32)
    test_data = np.reshape(test_data,[-1,1])
    test_label = np.loadtxt(test_label_file, delimiter=',',dtype=np.int32)
    test_label = np.reshape(test_label,[-1,1])

    num = math.floor(test_label.shape[0] / longth) * longth

    test_data = test_data[0:num,:]
    test_label = test_label[0:num,:]

    test_data = test_data.reshape((-1,longth,1))
    test_label = test_label.reshape((-1,longth))

    print(test_data.shape)
    print(test_label.shape)

    return train_data, train_label, test_data, test_label


def data_gen_final(train_data_file,train_label_file,longth):
    import math

    train_data = np.loadtxt(train_data_file, delimiter=',',dtype=np.float32)
    train_data = train_data.transpose()
    train_label = np.loadtxt(train_label_file, delimiter=',',dtype=np.int32)
    train_label = np.reshape(train_label,[-1,1])



    num = math.floor(train_label.shape[0] / longth) * longth

    train_data = train_data[0:num,:]
    train_label = train_label[0:num,:]

    train_data = train_data.reshape((-1,longth,65))
    train_label = train_label.reshape((-1,longth))

    return train_data, train_label







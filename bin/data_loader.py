### Libraries
import pickle
import gzip
import numpy as np
import pdb

def load_data():
    with gzip.open('D:/Git/NumberRecognition_DeepNeuralNetwork/bin/data/mnist.pkl.gz', 'rb') as fin:
        training_data, validation_data, test_data = pickle.load(fin, encoding='latin-1')
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    tr_d, va_d, tst_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorize(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in tst_d[0]]
    test_data = zip(test_inputs, tst_d[1])
    return (training_data, validation_data, test_data)

def vectorize(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

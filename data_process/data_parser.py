import logging
import numpy as np
import random


def parse_libsvm_data(line):
    try:
        line = line.decode('utf-8')
        splits = str.split(line.strip(), sep=' ')
        if splits[0] == '1' or splits[0] == '+1':
            label = float(1.0)
        else:
            label = float(0.0)

        splits = [str.split(elem, sep=':') for elem in splits[1:]]
        splits = [(int(elem[0]) - 1, float(elem[1])) for elem in splits]
        f_indices, f_values = zip(*splits)

        return label, f_indices, f_values
    except Exception as ex:
        logging.error('Error input: {}'.format(line))
        return float(-2.0), np.asarray([1]), np.asarray([1.0])


def gen_synthesis_data(line):
    if random.random() > 0.5:
        left = np.random.normal(0.5, 0.2, 5)
        right = np.random.normal(-0.5, 0.2, 5)
        f_values = np.concatenate((left, right), axis=0)
        return float(1.0), np.arange(10), f_values
    else:
        left = np.random.normal(0.5, 0.2, 5)
        right = np.random.normal(-0.5, 0.2, 5)
        f_values = np.concatenate((right, left), axis=0)
        return float(0.0), np.arange(10), f_values

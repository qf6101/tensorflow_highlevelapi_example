import tensorflow as tf
from data_process.data_set import read_dataset
from tensorflow.python import debug as tf_debug
import numpy as np
import logging


def read_data(config):
    sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': config.num_gpus}))
    if config.debug:
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)

    dataset = read_dataset(config)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    while True:
        try:
            elem = sess.run(next_element)

            features = elem[0]
            feature = features[0]
            label = elem[1]

            if not config.input_is_sparse and features.shape[1] != config.feature_size:
                logging.error(
                    'Wrong feature shape: {} (size of which should be {})'.format(features.shape, config.feature_size))

            print('feature type:', type(features))
            print('feature shape:', tuple(features.shape))
            print('label type:', type(label))
            print('label shape:', label.shape)
            print('first item\'s label:{}, f_indices:{}, f_values:{}'.format(label[0],
                                                                             np.where(feature != 0)[0][:5],
                                                                             list(feature[feature != 0][:5])))

        except tf.errors.OutOfRangeError:
            print("End of training dataset.")
            break

    sess.close()

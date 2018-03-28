import tensorflow as tf
from data_process.sparse_data_set import read_dataset
from tensorflow.python import debug as tf_debug
import numpy as np
from sklearn import linear_model
from scipy import sparse
import logging
from sklearn.externals import joblib
import os
import shutil
from sklearn.metrics import roc_auc_score


def train_sklearn_lr(config):
    if os.path.exists(config.model_dir):
        shutil.rmtree(config.model_dir)
    os.makedirs(config.model_dir)

    clf = linear_model.SGDClassifier(loss='log')

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
            labels = elem[1]
            f_indices = features.indices
            f_values = features.values

            if not config.input_is_sparse and features.shape[1] != config.feature_size:
                logging.error(
                    'Wrong feature shape: {} (size of which should be {})'.format(features.shape, config.feature_size))

            # print('feature type:', type(features))
            # print('feature shape:', tuple(features.dense_shape))
            # print('label type:', type(labels))
            # print('label shape:', labels.shape)
            # print('first item\'s label:{}, f_indices:{}, f_values:{}'.format(labels,
            #                                                                  list(tuple(i) for i in f_indices[:5]),
            #                                                                  f_values[:5]))

            X = sparse.csr_matrix((f_values, zip(*f_indices)), shape=(labels.shape[0], config.feature_size))
            Y = labels.ravel()
            clf.partial_fit(X, Y, classes=np.array([0, 1]))

        except tf.errors.OutOfRangeError:
            print("End of training dataset.")
            break

    sess.close()

    joblib.dump(clf, config.model_dir + '/model.pkl')


def eval_sklearn_lr(config):
    clf = joblib.load(config.model_dir + '/model.pkl')

    sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': config.num_gpus}))
    if config.debug:
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)

    trues = []
    predictions = []

    dataset = read_dataset(config)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    while True:
        try:
            elem = sess.run(next_element)
            features = elem[0]
            labels = elem[1]
            f_indices = features.indices
            f_values = features.values

            if not config.input_is_sparse and features.shape[1] != config.feature_size:
                logging.error(
                    'Wrong feature shape: {} (size of which should be {})'.format(features.shape, config.feature_size))

            X = sparse.csr_matrix((f_values, zip(*f_indices)), shape=(labels.shape[0], config.feature_size))
            Y = labels.ravel()

            scores = clf.predict_proba(X)

            # print('true: {}, score: {}'.format(list(Y), scores[:, 1]))

            trues.extend(list(Y))
            predictions.extend(scores[:, 1].ravel())

        except tf.errors.OutOfRangeError:
            print("End of training dataset.")
            break

    sess.close()

    auc = roc_auc_score(trues, predictions)
    print('auc: ', auc)

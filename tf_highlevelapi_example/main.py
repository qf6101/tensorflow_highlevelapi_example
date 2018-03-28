import tensorflow as tf
import argparse
from data_process.data_reader import read_data
from model.linear_model import train_linear_model, evaluate_linear_model, export_linear_model
from model.sklearn_model import train_sklearn_lr, eval_sklearn_lr

import logging.config
import logging
import sys


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()

parser.add_argument(
    '--task', type=str, default='read_data',
    help='task type (i.e., read_data, train, eval, export, train_sklearn_lr, eval_sklearn_lr)')

parser.add_argument(
    '--log_conf_file', type=str, default='conf/log.ini',
    help='path of log configuration file')

parser.add_argument(
    '--model_dir', type=str, default='/home/hzqianfeng/projects/remote/tf_highlevelapi_example/results/linear_model',
    help='path of model')

parser.add_argument(
    '--model_type', type=str, default='binary',
    help='model type (i.e., binary, direct, hidden)')

parser.add_argument(
    '--input_files', type=str,
    default='/data/qfeng/projects/tf_highlevelapi_example/data/a9a_pieces/*.gz',
    help='path of input files')

parser.add_argument(
    "--input_is_sparse", type=str2bool, nargs='?',
    const=True, default=True,
    help="Input data is sparse.")

parser.add_argument(
    "--gen_synthesis_data", type=str2bool, nargs='?',
    const=True, default=False,
    help="Generate synthesis data.")

parser.add_argument(
    "--use_sklearn_lr", type=str2bool, nargs='?',
    const=True, default=True,
    help="Use sklearn's logistic regression.")

parser.add_argument(
    '--export_dir', type=str,
    default='/home/hzqianfeng/projects/remote/tf_highlevelapi_example/results/linear_model_export',
    help='path of export files')

parser.add_argument(
    '--num_gpus', type=int, default=1,
    help='number of GPUs')

parser.add_argument(
    "--debug", type=str2bool, nargs='?',
    const=True, default=False,
    help="Activate debug mode.")

parser.add_argument(
    '--feature_size', type=int, default=123,
    help='total number of features')

parser.add_argument(
    '--label_size', type=int, default=2,
    help='total number of labels')

parser.add_argument(
    '--filenames_shuffle_buffer_size', type=int, default=100,
    help='buffer size for shuffling of file names')

parser.add_argument(
    '--num_epochs', type=int, default=1,
    help='number of epochs')

parser.add_argument(
    '--num_readers', type=int, default=2,
    help='number of readers')

parser.add_argument(
    '--shuffle_buffer_size', type=int, default=32,
    help='buffer size for shuffling of records')

parser.add_argument(
    '--num_parallel_calls', type=int, default=10,
    help='number of parallel calls')

parser.add_argument(
    '--batch_size', type=int, default=1,
    help='batch size')

parser.add_argument(
    '--prefetch_size', type=int, default=16,
    help='prefetch size')

FLAGS, unparsed_args = parser.parse_known_args()

print('=' * 50)
print('Command line Arguments:')
for arg in vars(FLAGS):
    print(arg, getattr(FLAGS, arg))
print('=' * 50)


def main(args):
    if FLAGS.task == 'read_data':
        read_data(FLAGS)
    elif FLAGS.task == 'train':
        train_linear_model(FLAGS)
    elif FLAGS.task == 'eval':
        evaluate_linear_model(FLAGS)
    elif FLAGS.task == 'export':
        export_linear_model(FLAGS)
    elif FLAGS.task == 'train_sklearn_lr':
        train_sklearn_lr(FLAGS)
    elif FLAGS.task == 'eval_sklearn_lr':
        eval_sklearn_lr(FLAGS)
    else:
        print('Wrong task.')


if __name__ == '__main__':
    logging.config.fileConfig(FLAGS.log_conf_file)
    if FLAGS.debug:
        tf.logging.set_verbosity(tf.logging.DEBUG)
    else:
        tf.logging.set_verbosity(tf.logging.INFO)
    logging.info('Start processing task {}'.format(FLAGS.task))
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed_args)
    logging.info('Finish processing task {}'.format(FLAGS.task))

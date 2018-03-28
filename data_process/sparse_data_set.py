import tensorflow as tf
import functools
from data_process.data_parser import parse_libsvm_data, gen_synthesis_data


def _compose_sparse_feature(label, f_indices, f_values, feature_size, label_size, is_binary_model):
    if not is_binary_model:
        label = tf.one_hot(indices=[label], depth=label_size)
    feature = tf.SparseTensor(tf.expand_dims(f_indices, axis=1), f_values, dense_shape=[feature_size])
    return feature, label


def read_dataset(config):
    import glob
    filenames = glob.glob(config.input_files)
    filename_dataset = tf.data.Dataset.from_tensor_slices(filenames)

    if config.filenames_shuffle_buffer_size > 0:
        filename_dataset = filename_dataset.shuffle(config.filenames_shuffle_buffer_size)

    filename_dataset = filename_dataset.repeat(config.num_epochs or None)

    records_dataset = filename_dataset.apply(
        tf.contrib.data.parallel_interleave(
            functools.partial(tf.data.TextLineDataset, compression_type='GZIP'),
            cycle_length=config.num_readers, sloppy=True))

    if config.shuffle_buffer_size > 0:
        records_dataset.shuffle(config.shuffle_buffer_size)

    if config.gen_synthesis_data:
        parse_data = gen_synthesis_data
    else:
        parse_data = parse_libsvm_data

    records_dataset = records_dataset.map(
        lambda line: tf.py_func(parse_data, [line], [tf.double, tf.int64, tf.double]),
        num_parallel_calls=config.num_parallel_calls)

    records_dataset = records_dataset.filter(lambda label, f_indices, f_values: label > -2.0)

    tensor_dataset = records_dataset.map(
        functools.partial(_compose_sparse_feature, feature_size=config.feature_size, label_size=config.label_size,
                          is_binary_model=(config.model_type == 'binary')),
        num_parallel_calls=config.num_parallel_calls)

    return tensor_dataset.batch(config.batch_size).prefetch(config.prefetch_size)

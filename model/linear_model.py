import tensorflow as tf
from data_process.data_set import read_dataset
import os
import shutil


def validate_batch_size_for_multi_gpu(num_gpus, batch_size):
    """For multi-gpu, batch-size must be a multiple of the number of
    available GPUs.

    Note that this should eventually be handled by replicate_model_fn
    directly. Multi-GPU support is currently experimental, however,
    so doing the work here until that feature is in place.
    """
    if batch_size > 0:
        if num_gpus > 4:
            from tensorflow.python.client import device_lib

            local_device_protos = device_lib.list_local_devices()
            num_gpus = sum([1 for d in local_device_protos if d.device_type == 'GPU'])
            if not num_gpus:
                raise ValueError('Multi-GPU mode was specified, but no GPUs '
                                 'were found. To use CPU, run without --multi_gpu.')

        remainder = batch_size % num_gpus
        if remainder:
            err = ('When running with multiple GPUs, batch size '
                   'must be a multiple of the number of available GPUs. '
                   'Found {} GPUs with a batch size of {}; try --batch_size={} instead.'
                   ).format(num_gpus, batch_size, batch_size - remainder)
            raise ValueError(err)


def input_fn(config):
    dataset = read_dataset(config)
    return dataset.make_one_shot_iterator().get_next()


def binary_model_fn(features, labels, mode, params):
    config = params['FLAGS']
    features = tf.reshape(features, (-1, config.feature_size))
    labels = tf.reshape(labels, (-1, 1))
    print('features shape: ', features.shape, ', labels shape: ', labels.shape)

    bias_reg = tf.contrib.layers.l2_regularizer(scale=0.1)
    kernel_reg = tf.contrib.layers.l2_regularizer(scale=0.1)

    predictions = tf.layers.dense(features, 1, activation=tf.nn.sigmoid, name='coeff',
                                  bias_regularizer=bias_reg, kernel_regularizer=kernel_reg)
    loss = tf.losses.log_loss(labels, predictions, reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        if config.num_gpus > 1:
            optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
    elif mode == tf.estimator.ModeKeys.EVAL:
        auc = tf.metrics.auc(labels, predictions)
        metrics = {'auc': auc}
        tf.summary.scalar('auc', auc[0])
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)


def direct_model_fn(features, labels, mode, params):
    config = params['FLAGS']
    features = tf.reshape(features, (-1, config.feature_size))
    labels = tf.reshape(labels, (-1, config.label_size))
    print('features shape: ', features.shape, ', labels shape: ', labels.shape)
    logits = tf.layers.dense(features, config.label_size, activation=tf.nn.sigmoid, name='coeff')
    print('logits shape: ', logits.shape)
    loss = tf.losses.softmax_cross_entropy(labels, logits, reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        if config.num_gpus > 1:
            optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def hidden_layer_model_fn(features, labels, mode, params):
    config = params['FLAGS']
    features = tf.reshape(features, (-1, config.feature_size))
    labels = tf.reshape(labels, (-1, config.label_size))
    print('features shape: ', features.shape, ', labels shape: ', labels.shape)
    logits1 = tf.layers.dense(features, 512, activation=tf.nn.relu, name='coeff1')
    logits2 = tf.layers.dense(logits1, config.label_size, activation=tf.nn.sigmoid, name='coeff2')
    print('logits shape: ', logits2.shape)
    loss = tf.losses.softmax_cross_entropy(labels, logits2, reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        if config.num_gpus > 1:
            optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def save_coefficients(classifier):
    bias = classifier.get_variable_value('coeff/bias')
    weights = classifier.get_variable_value('coeff/kernel')

    print('bias shape: {}'.format(bias.shape))
    print('weight shape: {}'.format(weights.shape))


def train_linear_model(config):
    if os.path.exists(config.model_dir):
        shutil.rmtree(config.model_dir)
    os.makedirs(config.model_dir)

    if config.model_type == 'binary':
        model_fn = binary_model_fn
    elif config.model_type == 'direct':
        model_fn = direct_model_fn
    elif config.model_type == 'hidden':
        model_fn = hidden_layer_model_fn
    else:
        raise Exception('Wrong model type.')

    if config.num_gpus > 1:
        validate_batch_size_for_multi_gpu(config.num_gpus, config.batch_size)
        model_fn_ = tf.contrib.estimator.replicate_model_fn(model_fn, loss_reduction=tf.losses.Reduction.MEAN)
    else:
        model_fn_ = model_fn

    classifier = tf.estimator.Estimator(
        model_fn=model_fn_,
        model_dir=config.model_dir,
        config=tf.estimator.RunConfig().replace(session_config=tf.ConfigProto(device_count={'GPU': config.num_gpus},
                                                                              allow_soft_placement=True)),
        params={
            'FLAGS': config
        }
    )

    classifier.train(input_fn=lambda: input_fn(config))


def evaluate_linear_model(config):
    if config.model_type == 'binary':
        model_fn = binary_model_fn
    elif config.model_type == 'direct':
        model_fn = direct_model_fn
    elif config.model_type == 'hidden':
        model_fn = hidden_layer_model_fn
    else:
        raise Exception('Wrong model type.')

    if config.num_gpus > 1:
        validate_batch_size_for_multi_gpu(config.num_gpus, config.batch_size)
        model_fn_ = tf.contrib.estimator.replicate_model_fn(model_fn, loss_reduction=tf.losses.Reduction.MEAN)
    else:
        model_fn_ = model_fn

    classifier = tf.estimator.Estimator(
        model_fn=model_fn_,
        model_dir=config.model_dir,
        config=tf.estimator.RunConfig().replace(session_config=tf.ConfigProto(device_count={'GPU': config.num_gpus},
                                                                              allow_soft_placement=True)),
        params={
            'FLAGS': config
        }
    )

    metrics = classifier.evaluate(input_fn=lambda: input_fn(config))
    print('auc: {}'.format(metrics['auc']))


def export_linear_model(config):
    if config.model_type == 'binary':
        model_fn = binary_model_fn
    elif config.model_type == 'direct':
        model_fn = direct_model_fn
    elif config.model_type == 'hidden':
        model_fn = hidden_layer_model_fn
    else:
        raise Exception('Wrong model type.')

    classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=config.model_dir,
        config=tf.estimator.RunConfig().replace(session_config=tf.ConfigProto(device_count={'GPU': 0})),
    )

    bias = classifier.get_variable_value('coeff/bias')
    weights = classifier.get_variable_value('coeff/kernel')

    print('bias', bias)
    print('weights', weights.ravel())


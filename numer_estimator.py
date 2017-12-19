"""A general tensorflow estimator for numerai models."""

import copy
import functools
import tensorflow as tf
from tensorflow.contrib.learn import ModeKeys
from tensorflow.python.estimator import estimator
from tensorflow.python.feature_column import feature_column

def _build_network(features):
  # Returns logits with shape [None, 1]
  feature_columns = [tf.feature_column.numeric_column(k)
                     for k in list(features)]
  input_layer = feature_column.input_layer(features, feature_columns)
  fc1 = tf.layers.dense(inputs=input_layer, units=1024,
                        activation=tf.nn.relu, name='fc1')
  fc2 = tf.layers.dense(inputs=fc1, units=512,
                        activation=tf.nn.relu, name='fc2')
  logits = tf.layers.dense(inputs=fc2, units=1, name='logits')
  #return tf.squeeze(logits, [1])
  return logits

def _build_train_op(loss, params):
  lr_decay_fn = functools.partial(
      tf.train.exponential_decay,
      decay_rate=0.7,
      decay_steps=1000,
      staircase=False)

  return tf.contrib.layers.optimize_loss(
      loss=loss,
      global_step=tf.train.get_global_step(),
      optimizer=tf.train.AdamOptimizer,
      learning_rate=params.learning_rate,
      learning_rate_decay_fn=lr_decay_fn)


def _build_eval_metric_ops(labels, predictions):
  return {
      'Accuracy': tf.metrics.accuracy(
          labels=labels,
          predictions=predictions,
          name='accuracy')
  }


def _maybe_expand_dims(tensor):
  if isinstance(tensor, tf.Tensor) and tensor.get_shape().ndims == 1:
    tensor = tf.expand_dims(tensor, -1)
  return tensor

def _numer_model_fn(features, labels, mode, params):
  is_training = mode == ModeKeys.TRAIN
  labels = _maybe_expand_dims(labels)
  logits = _build_network(features)
  predictions = tf.cast(tf.nn.sigmoid(logits) > 0.5, tf.int64)
  loss = tf.losses.sigmoid_cross_entropy(labels, logits)

  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=_build_train_op(loss, params),
      eval_metric_ops=_build_eval_metric_ops(labels, predictions))


class NumerEstimator(estimator.Estimator):

  def __init__(self, config=None, params=None):
    def _model_fn(features, labels, mode, params):
      return _numer_model_fn(
          features=features,
          labels=labels,
          mode=mode,
          params=params)
    super(NumerEstimator, self).__init__(
        model_fn=_model_fn, config=config, params=params)


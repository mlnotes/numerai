"""Main module to run numer models."""

import numer_estimator
import pandas as pd
import tensorflow as tf
from tensorflow.contrib.learn import learn_runner

def __process_data(training_file, eval_file):
  training_set = pd.read_csv(training_file, header=0)
  eval_set = pd.read_csv(eval_file, header=0)
  feature_names = [f for f in list(training_set) if 'feature' in f]
  return training_set, eval_set, feature_names

def __build_train_input_fn(data_set, feature_names):
  return tf.estimator.inputs.pandas_input_fn(
      x=data_set[feature_names],
      y=data_set['target'],
      num_epochs=None,
      shuffle=True)


def __build_eval_input_fn(data_set, feature_names):
  return tf.estimator.inputs.pandas_input_fn(
      x=data_set[feature_names],
      y=data_set['target'],
      num_epochs=1,
      shuffle=True)


def __experiment_fn(run_config, hparams):
  training_set, eval_set, feature_names = __process_data(
      training_file='data/numerai_training_data.csv',
      eval_file='data/numerai_tournament_data.csv')

  validation_set = eval_set.loc[eval_set.data_type == 'validation']
  test_set = eval_set.loc[eval_set.data_type == 'test']
  live_set = eval_set.loc[eval_set.data_type == 'live']

  #feature_columns = [tf.feature_column.numeric_column(k)
  #                   for k in feature_names]

  return tf.contrib.learn.Experiment(
      estimator=numer_estimator.NumerEstimator(run_config, hparams),
    #  estimator=tf.estimator.DNNClassifier(
    #      feature_columns=feature_columns,
    #      hidden_units=[1024, 512, 256],
    #      config=run_config,
    #  ),
      train_input_fn=__build_train_input_fn(training_set, feature_names),
      eval_input_fn=__build_eval_input_fn(validation_set, feature_names),
      train_steps=10001)


def main():
  config = tf.contrib.learn.RunConfig(
      model_dir='/usr/local/google/home/hanfeng/Desktop/tensorflow/numerai/logs',
      save_summary_steps=100,
      save_checkpoints_steps=100)

  params = tf.contrib.training.HParams(
      learning_rate=0.0005)

  learn_runner.run(experiment_fn=__experiment_fn,
                   run_config=config,
                   hparams=params,
                   schedule='train_and_evaluate')


if __name__ == '__main__':
  main()

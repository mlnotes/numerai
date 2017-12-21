"""Predicts with trained model."""

import numer_estimator
import tensorflow as tf
import pandas as pd

def main(_):
  data_set = pd.read_csv('/usr/local/google/home'
                         '/hanfeng/Desktop/tensorflow'
                         '/numerai/data/'
                         'numerai_tournament_data.csv.bak', header=0)
  feature_names = [f for f in list(data_set) if 'feature' in f]
  input_fn = tf.estimator.inputs.pandas_input_fn(
      x=data_set[feature_names],
      shuffle=False)



  config = tf.contrib.learn.RunConfig(
      model_dir='/tmp/numerai/logs')
  params = tf.contrib.training.HParams()
  estimator = numer_estimator.NumerEstimator(config, params)
  result = [i[0] for i in estimator.predict(input_fn)]

  result_set = pd.DataFrame(data={'probability': result})
  joined = pd.DataFrame(data_set.id).join(result_set)

  joined.to_csv('predictions.csv', index=False)


if __name__ == '__main__':
  tf.app.run()

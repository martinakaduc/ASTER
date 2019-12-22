import os
import logging
import tensorflow as tf
from PIL import Image
from google.protobuf import text_format
import numpy as np

from aster.protos import pipeline_pb2
from aster.builders import model_builder

# supress TF logging duplicates
logging.getLogger('tensorflow').propagate = False
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
logging.basicConfig(level=logging.INFO)

flags = tf.app.flags
flags.DEFINE_string('exp_dir', 'aster/experiments/demo/',
                    'Directory containing config, training log and evaluations')
flags.DEFINE_string('input_image', 'aster/data/10_ZCSU7023100 45G1_True_0.jpg', 'Demo image')
FLAGS = flags.FLAGS


def get_configs_from_exp_dir():
  pipeline_config_path = os.path.join(FLAGS.exp_dir, 'config/trainval.prototxt')

  pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
  with tf.compat.v1.gfile.GFile(pipeline_config_path, 'r') as f:
    text_format.Merge(f.read(), pipeline_config)

  model_config = pipeline_config.model
  eval_config = pipeline_config.eval_config
  input_config = pipeline_config.eval_input_reader

  return model_config, eval_config, input_config


def main(_):
  checkpoint_dir = os.path.join(FLAGS.exp_dir, 'log')
  # eval_dir = os.path.join(FLAGS.exp_dir, 'log/eval')
  model_config, _, _ = get_configs_from_exp_dir()

  model = model_builder.build(model_config, is_training=False)

  input_image_str_tensor = tf.compat.v1.placeholder(
    dtype=tf.string,
    shape=[])
  input_image_tensor = tf.image.decode_jpeg(
    input_image_str_tensor,
    channels=3,
  )
  resized_image_tensor = tf.image.resize(
    tf.cast(input_image_tensor, tf.float32),
    [64, 256])

  predictions_dict = model.predict(tf.expand_dims(resized_image_tensor, 0))
  recognitions = model.postprocess(predictions_dict)
  recognition_text = recognitions['text'][0]
  control_points = predictions_dict['control_points'],
  rectified_images = predictions_dict['rectified_images']

  saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
  checkpoint = os.path.join(FLAGS.exp_dir, 'log/model.ckpt')

  fetches = {
    'original_image': input_image_tensor,
    'recognition_text': recognition_text,
    'control_points': predictions_dict['control_points'],
    'rectified_images': predictions_dict['rectified_images'],
  }

  with open(FLAGS.input_image, 'rb') as f:
    input_image_str = f.read()

  with tf.compat.v1.Session() as sess:
    sess.run([
      tf.compat.v1.global_variables_initializer(),
      tf.compat.v1.local_variables_initializer(),
      tf.compat.v1.tables_initializer()])
    saver.restore(sess, checkpoint)
    sess_outputs = sess.run(fetches, feed_dict={input_image_str_tensor: input_image_str})

  print('Recognized text: {}'.format(sess_outputs['recognition_text'].decode('utf-8')))

  rectified_image = sess_outputs['rectified_images'][0]
  rectified_image_pil = Image.fromarray((128 * (rectified_image + 1.0)).astype(np.uint8))
  input_image_dir = os.path.dirname(FLAGS.input_image)
  rectified_image_save_path = os.path.join(input_image_dir, 'rectified_image.jpg')
  rectified_image_pil.save(rectified_image_save_path)
  print('Rectified image saved to {}'.format(rectified_image_save_path))

if __name__ == '__main__':
  tf.compat.v1.app.run()

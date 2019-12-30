import os
import logging
import tensorflow as tf
from PIL import Image
from google.protobuf import text_format
import numpy as np

from aster.protos import pipeline_pb2
from aster.builders import model_builder

class ASTER():
    def __init__(self):
        # supress TF logging duplicates
        logging.getLogger('tensorflow').propagate = False
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
        logging.basicConfig(level=logging.INFO)

        self.exp_dir = 'aster/experiments/demo/'
        tf.compat.v1.reset_default_graph()

        self.model_config, _, _ = self.get_configs_from_exp_dir()
        self.model = model_builder.build(self.model_config, is_training=False)

        self.input_image_str_tensor = tf.compat.v1.placeholder(
          dtype=tf.string,
          shape=[])

        self.input_image_tensor = tf.image.decode_jpeg(
          self.input_image_str_tensor,
          channels=3,
        )

        self.resized_image_tensor = tf.image.resize(
          tf.cast(self.input_image_tensor, tf.float32),
          [64, 256])

        self.predictions_dict = self.model.predict(tf.expand_dims(self.resized_image_tensor, 0))
        self.recognitions = self.model.postprocess(self.predictions_dict)
        self.recognition_text = self.recognitions['text'][0]
        self.reverse_text = self.recognitions['reverse']
        self.scores = self.recognitions['scores']
        self.control_points = self.predictions_dict['control_points'],
        self.rectified_images = self.predictions_dict['rectified_images']

        self.saver = tf.compat.v1.train.Saver()
        self.checkpoint = os.path.join(self.exp_dir, 'log/model.ckpt')

        self.fetches = {
          'original_image': self.input_image_tensor,
          'recognition_text': self.recognition_text,
          'reverse_text': self.reverse_text,
          'scores': self.scores,
          'control_points': self.predictions_dict['control_points'],
          'rectified_images': self.predictions_dict['rectified_images'],
        }

        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.compat.v1.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        self.sess.run([
          tf.compat.v1.global_variables_initializer(),
          tf.compat.v1.local_variables_initializer(),
          tf.compat.v1.tables_initializer()])

        print('LOADING ASTER MODEL...')
        self.saver.restore(self.sess, self.checkpoint)
        print('ASTER MODEL READY!')

    def get_configs_from_exp_dir(self):
      pipeline_config_path = os.path.join(self.exp_dir, 'config/trainval.prototxt')

      pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
      with tf.compat.v1.gfile.GFile(pipeline_config_path, 'r') as f:
        text_format.Merge(f.read(), pipeline_config)

      model_config = pipeline_config.model
      eval_config = pipeline_config.eval_config
      input_config = pipeline_config.eval_input_reader

      return model_config, eval_config, input_config

    def process(self, image_folder='./text_image'):
      result = []

      image_name_list = os.listdir(image_folder)
      image_name_list = [x for x in image_name_list if x[-3:].lower() == 'jpg' or x[-3:].lower() == 'png']

      for input_image in image_name_list:
          with open(image_folder + '/' + input_image, 'rb') as f:
            input_image_str = f.read()
          sess_outputs = self.sess.run(self.fetches, feed_dict={self.input_image_str_tensor: input_image_str})

          text_result = sess_outputs['recognition_text'].decode('utf-8')
          if sess_outputs['reverse_text'][0]:
              text_result = text_result[::-1]
          # print('Recognized text: {}'.format(text_result))
          result.append([input_image, text_result.upper(), 1+sess_outputs['scores'][0]])
      return result

if __name__ == '__main__':
    aster = ASTER()
    res = aster.process('aster/data')
    print(res)

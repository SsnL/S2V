# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Train the skip-thoughts model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import numpy as np
import torch
import json
from tqdm.auto import tqdm

import configuration
import s2v_encoder
import s2v_model

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("input_file_pattern", None,
                       "File pattern of sharded TFRecord input files.")
tf.flags.DEFINE_string("results_path", None, "Model results path")
tf.flags.DEFINE_string("eval_dir", None, "Directory to write event logs to.")
tf.flags.DEFINE_string("master", "", "Eval master.")
tf.flags.DEFINE_float("uniform_init_scale", 0.1, "Random init scale")
tf.flags.DEFINE_boolean("shuffle_input_data", False, "Whether to shuffle data")
tf.flags.DEFINE_integer("input_queue_capacity", 640000, "Input data queue capacity")
tf.flags.DEFINE_integer("num_input_reader_threads", 1, "Input data reader threads")
tf.flags.DEFINE_integer("num_eval_examples", 50000,
                        "Number of examples for evaluation.")
tf.flags.DEFINE_integer("batch_size", 400, "Batch size")
tf.flags.DEFINE_integer("sequence_length", 30, "Max sentence length considered")
tf.flags.DEFINE_integer("context_size", 1, "Prediction context size")
tf.flags.DEFINE_boolean("dropout", False, "Use dropout")
tf.flags.DEFINE_float("dropout_rate", 0.3, "Dropout rate")
tf.flags.DEFINE_string("model_config", None, "Model configuration json")

def main(unused_argv):
  os.makedirs(FLAGS.eval_dir, exist_ok=True)

  if not os.path.exists(os.path.join(FLAGS.eval_dir, '../train/model.ckpt-131998.data-00000-of-00001')):
    torch.save('INF_OR_NAN', os.path.join(FLAGS.eval_dir, 'INF_OR_NAN'))
    print('INF_OR_NAN\n\n\n\n\n')
    return

  assert FLAGS.num_eval_examples % FLAGS.batch_size == 0

  if not FLAGS.input_file_pattern:
    raise ValueError("--input_file_pattern is required.")
  if not FLAGS.results_path:
    raise ValueError("--results_path is required.")
  if not FLAGS.eval_dir:
    raise ValueError("--eval_dir is required.")

  eval_dir = FLAGS.eval_dir
  if not tf.gfile.IsDirectory(eval_dir):
    tf.logging.info("Creating eval directory: %s", eval_dir)
    tf.gfile.MakeDirs(eval_dir)

  with open(FLAGS.model_config) as json_config_file:
    json_model_config = json.load(json_config_file)
  # import pdb; pdb.set_trace()
  model_config = configuration.model_config(json_model_config, mode="eval")

  g = tf.Graph()
  with g.as_default():
    encoder = s2v_encoder.s2v_encoder(model_config)
    restore_model = encoder.build_graph_from_config(model_config, mode="eval")

  sess = tf.Session(graph=g)
  restore_model(sess)

  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)

  encs_f = []
  encs_g = []

  for _ in tqdm(range(FLAGS.num_eval_examples // FLAGS.batch_size)):
    enc_f, enc_g = sess.run(
      ["encoder/thought_vectors:0", "encoder_out/thought_vectors:0"],
            feed_dict={})
    encs_f.append(enc_f)
    encs_g.append(enc_g)

  coord.request_stop()

  # Wait for threads to finish.
  coord.join(threads)
  sess.close()

  encs_f = torch.as_tensor(np.concatenate(encs_f, 0))
  encs_g = torch.as_tensor(np.concatenate(encs_g, 0))

  assert encs_f.shape == encs_g.shape
  assert encs_f.shape[0] == FLAGS.num_eval_examples

  # save_file = os.path.join(FLAGS.eval_dir, 'val_encs.pth')
  # torch.save((encs_f, encs_g), save_file)
  save_file = os.path.join(FLAGS.eval_dir, 'val_encs.pth.dict')
  torch.save(dict(fs=encs_f, gs=encs_g), save_file)
  print(f'saved to {save_file}\n\n\n\n\n')

if __name__ == "__main__":
  tf.app.run()

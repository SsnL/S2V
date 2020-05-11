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
"""

The evaluation tasks have different running times. SICK may take 5-10 minutes.
MSRP, TREC and CR may take 20-60 minutes. SUBJ, MPQA and MR may take 2+ hours.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import warnings

import eval_classification
import eval_msrp
import eval_sick
import eval_trec

import tensorflow as tf
import numpy as np
import torch

import configuration
import encoder_manager

import json

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("eval_task", "MSRP",
                       "Name of the evaluation task to run. Available tasks: "
                       "MR, CR, SUBJ, MPQA, SICK, MSRP, TREC.")
tf.flags.DEFINE_string("eval_dir", None, "Directory to write results to.")
tf.flags.DEFINE_string("data_dir", None, "Directory containing training data.")
tf.flags.DEFINE_float("uniform_init_scale", 0.1, "Random init scale")
tf.flags.DEFINE_integer("batch_size", 400, "Batch size")
tf.flags.DEFINE_boolean("use_norm", False,
                        "Normalize sentence embeddings during evaluation")
tf.flags.DEFINE_integer("sequence_length", 30, "Max sentence length considered")
tf.flags.DEFINE_string("model_config", None, "Model configuration json")
tf.flags.DEFINE_string("results_path", None, "Model results path")
tf.flags.DEFINE_string("Glove_path", None, "Path to Glove dictionary")

tf.logging.set_verbosity(tf.logging.INFO)

def main(unused_argv):
  os.makedirs(FLAGS.eval_dir, exist_ok=True)

  if not os.path.exists(os.path.join(FLAGS.eval_dir, '../train/model.ckpt-131998.data-00000-of-00001')):
    print('ABORT: INF_OR_NAN\n\n\n\n\n')
    return

  if not FLAGS.data_dir:
    raise ValueError("--data_dir is required.")
  if not FLAGS.model_config:
    raise ValueError("--model_config is required.")

  encoder = encoder_manager.EncoderManager(allow_growth=True)

  with open(FLAGS.model_config) as json_config_file:
    json_model_configs = json.load(json_config_file)

  if type(json_model_configs) is dict:
    json_model_configs = [json_model_configs]

  for json_mdl_cfg in json_model_configs:
    model_config = configuration.model_config(json_mdl_cfg, mode="encode")
    encoder.load_model(model_config)

  eval_tasks = [t for t in re.split(r'_|\+', FLAGS.eval_task) if len(t) > 0]

  save_results_file = os.path.join(FLAGS.eval_dir, 'eval_results_new.pth')
  if os.path.exists(save_results_file):
    eval_results = torch.load(save_results_file)
  else:
    eval_results = {}

  for eval_task in eval_tasks:
    print(f"\n\n\nrunning {eval_task}...\n")
    if eval_task in eval_results:
      warnings.warn(f"{eval_task} already exists, skip: current keys = {eval_results.keys()}")
      # if not isinstance(eval_results[eval_task], dict):
      continue

    if eval_task in ["MR", "CR", "SUBJ", "MPQA"]:
      data_dir = os.path.join(FLAGS.data_dir, eval_task)
      results = eval_classification.eval_nested_kfold(
          encoder, eval_task, data_dir, use_nb=False, just_one=True)
      assert len(results['train']) == len(results['test']) == 1
      print('Train score', np.mean(results['train']))
      print('Test score', np.mean(results['test']))
      results = dict(train= np.mean(results['train']), test=np.mean(results['test']))
    elif eval_task == "SICK":
      results = eval_sick.evaluate(encoder, evaltest=True, loc=data_dir)
    elif eval_task == "MSRP":
      results = eval_msrp.evaluate(
          encoder, evalcv=True, evaltest=True, use_feats=False, loc=data_dir)
    elif eval_task == "TREC":
      eval_trec.evaluate(encoder, evalcv=True, evaltest=True, loc=data_dir)
    else:
      raise ValueError("Unrecognized eval_task: %s" % eval_task)

    eval_results[eval_task] = results
  encoder.close()

  torch.save(eval_results, save_results_file)
  print(f'saved to {save_results_file}\n\n\n\n')

if __name__ == "__main__":
  tf.app.run()

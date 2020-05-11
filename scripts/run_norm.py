import os
import sys
import glob
import tempfile
import subprocess
import pickle
import json


ROOT = '/data/vision/torralba/distillation/S2V/'
# IMAGE_ROOT = "/data/vision/torralba/scratch2/stevenliu/GAN_stability/fid_samples"


def train_conf_template(loss_config):
    return r"""
{
  "encoder": "gru",
  "encoder_dim": 1200,
  "encoder_norm": true,
  "bidir": true,
  "checkpoint_path": "",
  "loss_config": """ + json.dumps(loss_config) + r""",
  "vocab_configs": [
  {
      "mode": "trained",
      "name": "word_embedding",
      "dim": 620,
      "size": 50001,
      "vocab_file": "dictionaries/BC/dictionary.txt",
      "embs_file": ""
  }
  ]
}
"""

def eval_conf_template(loss_config, desc):
    return r"""
{
  "encoder": "gru",
  "encoder_dim": 1200,
  "encoder_norm": true,
  "bidir": true,
  "case_sensitive": true,
  "checkpoint_path": "BS400-W620-S1200-case-bidir-norm/""" + desc + """/train",
  "loss_config": """ + json.dumps(loss_config) + r""",
  "vocab_configs": [
  {
    "mode": "trained",
    "name": "word_embedding",
    "cap": false,
    "dim": 620,
    "size": 50001,
    "vocab_file": "dictionaries/BC/dictionary.txt",
    "embs_file": ""
  }
  ]
}
"""

def get_extra_flag(mode):
  mode = mode.upper()
  EXTRA_FLAG = os.environ.get(f"EXTRA_{mode}_FLAG", None)
  if EXTRA_FLAG is None:
    return []
  else:
    return EXTRA_FLAG.split(' ')


def write_confs_get_cmds(loss_config, desc, BS=400, SEQ_LEN=30, NUM_INST=52799513, GLOVE_PATH="dictionaries/GloVe", RESULTS_HOME='results/BC'):
    conf_base_dir = os.path.join(ROOT, 'model_configs', 'BS400-W620-S1200-case-bidir-norm', desc)
    os.makedirs(conf_base_dir, exist_ok=True)
    with open(os.path.join(conf_base_dir, 'train.json'), 'w') as f:
        f.write(train_conf_template(loss_config=loss_config))
    with open(os.path.join(conf_base_dir, 'eval.json'), 'w') as f:
        f.write(eval_conf_template(loss_config=loss_config, desc=desc))

    results_base_dir = os.path.join(RESULTS_HOME, 'BS400-W620-S1200-case-bidir-norm', desc)
    os.makedirs(results_base_dir, exist_ok=True)
    train_cmd = [
        sys.executable, '-u', 'src/train.py',
        f'--input_file_pattern=TFRecords/BC_filter_empty/train-?????-of-00100',
        f'--train_dir={results_base_dir}/train',
        f'--learning_rate_decay_factor=0',
        f'--batch_size={BS}',
        f'--sequence_length={SEQ_LEN}',
        f'--nepochs=1',
        f'--num_train_inst={NUM_INST}',
        f'--save_model_secs=1800',
        f'--Glove_path={GLOVE_PATH}',
        f'--model_config={conf_base_dir}/train.json'
    ] + get_extra_flag('train')

    eval_cmd = [
        sys.executable, '-u', 'src/eval.py',
        f'--input_file_pattern=TFRecords/BC_filter_empty/validation-?????-of-00001',
        f'--checkpoint_dir={results_base_dir}/train',
        f'--eval_dir={results_base_dir}/eval',
        f'--batch_size={BS}',
        f'--sequence_length={SEQ_LEN}',
        f'--model_config={conf_base_dir}/train.json',
        f'--eval_interval_secs=1800',
    ] + get_extra_flag('eval')

    return [train_cmd]#, eval_cmd

def run_cmd(cmd):
    with subprocess.Popen(cmd,
                          stdout=subprocess.PIPE, cwd=ROOT,
                          bufsize=1, universal_newlines=True,
                          encoding='utf-8') as p:
        for line in p.stdout:
            print(line, end='', flush=True)

    if p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, p.args)


if __name__ == '__main__':
    loss_config = {}
    terms = []
    for arg in sys.argv[1:]:
        terms += list(arg.split('_'))
    for term in terms:
        i = 0
        while term[i].isalpha():
            i += 1
        leading = term[:i]
        while i < len(term):
            j = i
            while term[j].isalpha():
                j += 1
            name = leading + term[i:j]
            i = j
            while j < len(term) and not term[j].isalpha():
                j += 1
            val = float(term[i:j])
            loss_config[name] = val
            i = j

    print(loss_config)

    desc = '_'.join(sys.argv[1:])

    for cmd in write_confs_get_cmds(loss_config, desc):
        print()
        print()
        print(' \\\n\t'.join(cmd))
        print()
        print()
        run_cmd(cmd)



import os
import sys
import glob
import tempfile
import subprocess
import pickle
import json


ROOT = '/data/vision/torralba/distillation/S2V/'
# IMAGE_ROOT = "/data/vision/torralba/scratch2/stevenliu/GAN_stability/fid_samples"



def get_cmd(desc, BS=400, TASK='MR+CR+SUBJ+MPQA', GLOVE_PATH="dictionaries/GloVe", ST_DATA='./ST_data', RESULTS_HOME='results/BC'):
    return [
        sys.executable, '-u', 'src/evaluate.py',
          f'--eval_task={TASK}',
          f'--data_dir={ST_DATA}',
          f'--model_config=model_configs/BS400-W620-S1200-case-bidir-norm/{desc}/eval.json',
          f'--results_path=results/BC',
          f'--eval_dir=results/BC/BS400-W620-S1200-case-bidir-norm/{desc}/eval',
          f'--Glove_path={GLOVE_PATH}',
    ]

    # python src/evaluate.py \
    #     --eval_task=MR+CR \
    #     --data_dir=./ST_data \
    #     --model_config='model_configs/BS400-W620-S1200-case-bidir/eval.json' \
    #     --results_path='results/BC' \
    #     --eval_dir='results/BC/BS400-W620-S1200-case-bidir/eval' \
    #     --Glove_path='dictionaries/GloVe'

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
    desc = sys.argv[1]
    TASK = os.environ.get('TASK', 'MR+CR')
    cmd = get_cmd(desc, TASK=TASK)
    print()
    print()
    print(' \\\n\t'.join(cmd))
    print()
    print()
    run_cmd(cmd)



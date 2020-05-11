import os
import sys
import base64
import socket
import glob
import tempfile
import subprocess
import pickle
import json
import fcntl

LOCK_DIR = '/afs/csail.mit.edu/u/t/tongzhou/.filelocks'
LOCK_SUFFIX = '.lock'
SUCCESS_SUFFIX = '.claimed'
EXTRA_DESC = os.environ.get('EXTRA_DESC', None)

def get_lock_subdir(script, list_files):
    def encode_path(path):
        realpath = os.path.realpath(path)
        return os.path.basename(path) + '__' + base64.b64encode(realpath.encode('utf-8')).decode()

    parts = [encode_path(list_file) for list_file in list_files]

    if EXTRA_DESC is not None:
        parts.append(EXTRA_DESC)

    return os.path.join(LOCK_DIR, encode_path(script), '___'.join(parts))

def try_get_lock(lock_subdir, item, cuda_visible_devices, runner_desc):
    success_file = os.path.join(lock_subdir, item + SUCCESS_SUFFIX)
    os.makedirs(os.path.split(success_file)[0], exist_ok=True)  # in case item contains /
    if os.path.exists(success_file):
        return False, None
    lock_file = os.path.join(lock_subdir, item + LOCK_SUFFIX)
    with open(lock_file, 'w') as f:
        try:
            fcntl.lockf(f, fcntl.LOCK_NB | fcntl.LOCK_EX)
        except BlockingIOError:
            return False, None
    with open(success_file, 'w') as f:
        print(f'RUNNER HOST: {socket.gethostname()}', file=f)
        print(f'RUNNER PID: {os.getpid()}', file=f)
        print(f'RUNNER CUDA_VISIBLE_DEVICES: {cuda_visible_devices}', file=f)
        print(f'RUNNER DESC: {runner_desc}', file=f)
    def callback(exitcode):
        with open(success_file, 'a') as f:
            print(f'RUNNER EXITCODE: {exitcode}', file=f)
    return True, callback

def run_cmd(cmd):
    with subprocess.Popen(cmd,
                          stdout=subprocess.PIPE,
                          bufsize=1, universal_newlines=True,
                          encoding='utf-8') as p:
        for line in p.stdout:
            print(line, end='', flush=True)

    if p.returncode != 0:
        print(f'\n\nWARN: Subprocess exited with {p.returncode}.\n')

    return p.returncode


if __name__ == '__main__':
    script = sys.argv[1]
    list_files = sys.argv[2:]

    items = []

    for list_file in list_files:
        with open(list_file, 'r') as f:
            for line in f.readlines():
                item = line.strip()
                if len(item):
                    assert item not in items
                    items.append(item)

    lock_subdir = get_lock_subdir(script, list_files)
    os.makedirs(lock_subdir, exist_ok=True)
    print(f"\n\nLOCK DIR: {lock_subdir}\n\n")

    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'UNSET')
    runner_desc = os.environ.get('RUNNER_DESC', None)

    for item in items:
        got_lock, callback = try_get_lock(lock_subdir, item, cuda_visible_devices=cuda_visible_devices, runner_desc=runner_desc)
        if got_lock:
            print(f'\n\n============================\n !!!GOT LOCK FOR {item}\n============================\n\n')
            callback(run_cmd([sys.executable, '-u', script, item]))
        else:
            print(f'\n\n============================\n :( NO LOCK FOR {item}\n============================\n\n')


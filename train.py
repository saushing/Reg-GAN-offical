#!/usr/bin/python3

import argparse
import os
from trainer import Cyc_Trainer
import yaml
import subprocess
import webbrowser
import time

#os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.safe_load(stream)
    
def launch_tensorboard(logdir, port=6006, wait_sec=3):
    """
    Launch TensorBoard and open browser automatically.
    """
    print(f"[INFO] Launching TensorBoard at {logdir}")

    # Start TensorBoard as a subprocess
    tb_cmd = [
        "tensorboard",
        "serve",
        "--logdir", logdir,
        "--port", str(port)
    ]

    subprocess.Popen(
        tb_cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        shell=(os.name == "nt")  # Windows needs shell=True
    )

    # Wait a bit for server to start
    time.sleep(wait_sec)

    url = f"http://127.0.0.1:{port}"
    print(f"[INFO] Opening TensorBoard in browser: {url}")
    webbrowser.open(url)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='Yaml/CycleGan.yaml', help='Path to the config file.')
    opts = parser.parse_args()
    config = get_config(opts.config)
    
    if config['name'] == 'CycleGan':
        trainer = Cyc_Trainer(config)
    
    logdir = os.path.join('runs', config['name'])
    launch_tensorboard(logdir)
    trainer.train()

###################################
if __name__ == '__main__':
    main()
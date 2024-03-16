from distutils.command.config import config

import os
import shutil
import random
import yaml
import argparse
import numpy as np
import pandas as pd

def get_data_deep(config_file):
    config=read_params_deep(config_file)
    return config

def read_params_deep(config_file):
    with open(config_file) as conf:
        config=yaml.safe_load(conf)
    return config


if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config",default="deep_params.yaml")
    parsed_args=args.parse_args()
    data = get_data_deep(config_file=parsed_args.config)
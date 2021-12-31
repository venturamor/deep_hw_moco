# parser --------------------------------------------------------------------------
import argparse
import yaml

parser = argparse.ArgumentParser(description='Configuration to MOCO model and Classifier')
parser.add_argument('--config', default='config_moco.yaml', type=str,
                    help='Path to yaml config file. defualt: config_moco.yaml')
args = parser.parse_args()
with open(args.config, encoding="utf8") as f:
    global config_args
    config_args = yaml.load(f, Loader=yaml.FullLoader)

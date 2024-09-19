import json
import argparse
from trainer import train


def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    args.update(param)  # Add parameters from json

    train(args)


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)

    return param


def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
    parser.add_argument('--config', type=str, default='./exps/ours.json',
                        help='Json file of settings.')
    parser.add_argument('--ablation', type=str, default='proposed',
                        help='option: [proposed, no_aug, use_radius, no_contrastive]')
    parser.add_argument('--drawing', action='store_true')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=512)
    
    return parser


if __name__ == '__main__':
    main()

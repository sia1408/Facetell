import argparse
import yaml
from src.train import train_model
from src.evaluate import evaluate_model

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    parser = argparse.ArgumentParser(description="DeepFake Detection Model")
    parser.add_argument('--mode', choices=['train', 'evaluate'], required=True,
                        help='Mode to run the script in: train or evaluate')
    parser.add_argument('--config', default='config.yaml', help='Path to the configuration file')
    args = parser.parse_args()

    config = load_config(args.config)

    if args.mode == 'train':
        train_model(config)
    elif args.mode == 'evaluate':
        evaluate_model(config)

if __name__ == "__main__":
    main()
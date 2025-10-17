import os
import sys
sys.path.append(os.path.join(os.path.abspath(__file__), 'src'))

import argparse
import json

from dotenv import load_dotenv
load_dotenv()

from trainer import Conv1dAETrainer, load_ae_model

def parse_args() -> dict:
    parser = argparse.ArgumentParser(description="Train Conv1d Autoencoder")
    
    parser.add_argument('--preset', type=str, choices=['minimum', 'dev', 'prod'], required=False, default='prod', help='Preset configuration to use.')
    parser.add_argument('--data_path', type=str, required=False, default='csv/board_snapshots.csv', help='Path to the training data CSV file')
    parser.add_argument('--epochs', type=int, required=False, default=15, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, required=False, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, required=False, default=1e-3, help='Learning rate for optimizer')
    parser.add_argument('--mode', type=str, choices=['train_and_eval', 'train', 'eval'], required=False, default='train_and_eval', help='Mode: train or evaluate.')
    
    args = parser.parse_args()
    
    with open(os.path.join(os.path.dirname(__file__), 'presets.json'), 'r') as f:
        preset = json.load(f).get(args.preset, {})
    
    args = vars(args) | preset
    
    print("Using configuration:")
    for key, value in args.items():
        print(f"  {key}: {value}")
    
    return args

def main() -> None:
    args = parse_args()
    mode = args.get('mode', 'train_and_eval')
    model_path = os.getenv('MODEL_PATH', 'models')
    effective_depth_level = int(os.getenv('EFFECTIVE_DEPTH_LEVEL', 10))
    
    trainer = Conv1dAETrainer(**args, effective_depth_level=effective_depth_level, model_path=model_path)
    
    trained_model = None
    
    if mode in ['train', 'train_and_eval']:
        trained_model = trainer.train()
    
    if mode in ['eval', 'train_and_eval']:
        if trained_model is None:
            trained_model = load_ae_model(model_path)
        
        trainer.evaluate(trained_model)

if __name__ == "__main__":
    main()
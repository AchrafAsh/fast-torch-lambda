import argparse
import tqdm
import time

from models.train import run


parser = argparse.ArgumentParser(prog="Trainer", allow_abbrev=False)

# parser.add_argument("--train", action='store_true')
parser.add_argument("--epochs", type=int, help="Number of epochs to train", required=False, default=2)
parser.add_argument("--model", type=str, help="Path to the save model (with .pt extension)")
parser.add_argument("--batch", type=int, help="Batch size", default=12)

args = parser.parse_args()

run(batch_size=args.batch, epochs=args.epochs, model_path=args.model)
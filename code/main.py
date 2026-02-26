import argparse
import torch
import numpy as np
import random
from tqdm import tqdm

from data import get_dataloaders
from model import Restricted_Transformer
from train_test import train, test


def set_model_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_data_seed(seed:int):
    random.seed(seed)
    np.random.seed(seed)

def main():
    parser = argparse.ArgumentParser(
        description="Tiny attention classifier – majority or 'at-least-one 1' (OR) task",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--task",
        choices=["majority", "or"],
        default="majority",
        help="Task to run: 'majority' = more than half are 1s | 'or' = at least one 1 appears"
    )
    parser.add_argument("--epochs", type=int, default=25,
                        help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-num-samples", type=int, default=8192)
    parser.add_argument("--train-seq-len", type=int, default=31)
    parser.add_argument("--test-num-samples", type=int, default=2048)
    parser.add_argument("--test-seq-len", type=int, default=31)
    parser.add_argument(
        "--train-saturated",
        action="store_true",
        default=False,
        help="Train with saturated (hard argmax) attention instead of softmax"
    )
    parser.add_argument("--optim", choices=["AdamW", "SGD"], default="AdamW")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--data_seed", type = int, default=42)
    parser.add_argument("--device", type=str, default=None,
                        help="cuda / cpu / mps — auto-detected if not set")

    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Task:  {args.task.upper()}\n")

    set_data_seed(args.data_seed)
    is_majority = (args.task == "majority")
    
    train_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        train_num_samples=args.train_num_samples,
        train_seq_len=args.train_seq_len,
        test_num_samples=args.test_num_samples,
        test_seq_len=args.test_seq_len,
        majority=is_majority,
        )
    
    results = {}

    for seed in tqdm(range(10),desc = "should take a minute :)"):  
        set_model_seed(seed)
        print(f"Seed {seed} ────────────────────────────────────────")

        model = Restricted_Transformer(emb_size=1).to(device)

        if args.optim == "AdamW":
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

        loss_fn = torch.nn.BCEWithLogitsLoss()

        for epoch in range(1, args.epochs + 1):
            model.attention.saturated = args.train_saturated
            loss, _ = train(model, train_loader, loss_fn, optimizer, device)

            if epoch % 5 == 0 or epoch == args.epochs:
                print(f"  ep {epoch:3d} | loss {loss:.5f}")

        # Final eval – both modes
        model.attention.saturated = False
        acc_soft = test(model, test_loader, device)

        model.attention.saturated = True
        acc_hard = test(model, test_loader, device)

        results[seed] = {
            "softmax_acc": round(acc_soft, 2),
            "saturated_acc": round(acc_hard, 2),
        }

        print(f"  softmax    : {acc_soft:5.2f}%")
        print(f"  saturated  : {acc_hard:5.2f}%\n")

    print(results)


if __name__ == "__main__":
    main()

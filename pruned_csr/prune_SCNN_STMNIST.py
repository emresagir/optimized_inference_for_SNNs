"""
Iterative magnitude pruning + fine-tuning of the trained ST-MNIST SCNN.

Loads the dense checkpoint (same architecture as train_SCNN_STMNIST.py),
then for each target sparsity level:
    1. Global unstructured L1 pruning over conv1, conv2, fc (bias=False everywhere)
    2. Fine-tune with the pruning masks active (pruned weights stay exactly 0)
    3. Keep the best epoch by validation accuracy
    4. Make pruning permanent (prune.remove) on a copy and save a plain
       state_dict ('0.weight', '2.weight', '5.weight'), so it loads into the
       unchanged nn.Sequential and exports through the usual snntorch -> NIR path.

Levels are cumulative: each level starts from the fine-tuned previous level.

Usage:
    python3 prune_SCNN_STMNIST.py
    python3 prune_SCNN_STMNIST.py --sparsities 0.2 0.4 0.5 0.7 --finetune_epochs 5 --lr 1e-3
"""

import argparse
import copy
import json
import os
import random
import time
from datetime import datetime

# For determinism (must be set before CUDA init)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader

import tonic
import snntorch as snn
from snntorch import functional as SF
from snntorch import utils


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_ROOT = os.path.join(SCRIPT_DIR, "..", "SCNN_STMNIST", "cache", "stmnist")

# Must match training
BETA = 0.95
RESET_MECHANISM = "subtract"
BATCH_SIZE = 32

# Indices of the prunable layers inside the nn.Sequential
PRUNABLE = {"conv1": 0, "conv2": 2, "fc": 5}


# ============================================================
# Setup helpers
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_net(device):
    """Exact architecture from train_SCNN_STMNIST.py."""
    return nn.Sequential(
        nn.Conv2d(2, 8, kernel_size=4, stride=1, bias=False),   # 10x10x2 -> 7x7x8
        snn.Leaky(beta=BETA, init_hidden=True, reset_mechanism=RESET_MECHANISM),
        nn.Conv2d(8, 16, kernel_size=3, stride=2, bias=False),  # 7x7x8 -> 3x3x16
        snn.Leaky(beta=BETA, init_hidden=True, reset_mechanism=RESET_MECHANISM),
        nn.Flatten(),
        nn.Linear(16 * 3 * 3, 10, bias=False),
        snn.Leaky(beta=BETA, init_hidden=True, output=True, reset_mechanism=RESET_MECHANISM),
    ).to(device)


def build_loaders(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    collate = tonic.collation.PadTensors(batch_first=False)

    def ds(split):
        return tonic.DiskCachedDataset(dataset=None, cache_path=os.path.join(CACHE_ROOT, split))

    trainloader = DataLoader(ds("train"), batch_size=BATCH_SIZE, collate_fn=collate,
                             shuffle=True, worker_init_fn=seed_worker, generator=g)
    valloader = DataLoader(ds("val"), batch_size=BATCH_SIZE, collate_fn=collate,
                           worker_init_fn=seed_worker, generator=g)
    testloader = DataLoader(ds("test"), batch_size=BATCH_SIZE, collate_fn=collate,
                            worker_init_fn=seed_worker, generator=g)
    return trainloader, valloader, testloader


def forward_pass(net, data):
    spk_rec = []
    utils.reset(net)
    for step in range(data.size(0)):
        spk_out, _ = net(data[step])
        spk_rec.append(spk_out)
    return torch.stack(spk_rec)


@torch.no_grad()
def evaluate(net, loader, device):
    """Exact sample-level accuracy (spike-count argmax), not a mean of batch means."""
    net.eval()
    correct, total = 0, 0
    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        spk_rec = forward_pass(net, data)
        pred = spk_rec.sum(dim=0).argmax(dim=1)
        correct += (pred == targets).sum().item()
        total += targets.numel()
    return correct / total


# ============================================================
# Pruning helpers
# ============================================================

def prune_targets(net, layer_names):
    return [(net[PRUNABLE[name]], "weight") for name in layer_names]


def layer_sparsity_report(net, layer_names):
    """Works both with active masks (module.weight is the masked tensor) and after prune.remove."""
    report, zeros_total, numel_total = {}, 0, 0
    for name in layer_names:
        w = net[PRUNABLE[name]].weight
        zeros = int((w == 0).sum().item())
        numel = w.numel()
        report[name] = {"zeros": zeros, "numel": numel, "sparsity": zeros / numel}
        zeros_total += zeros
        numel_total += numel
    report["global"] = {"zeros": zeros_total, "numel": numel_total,
                        "sparsity": zeros_total / numel_total}
    return report


def print_sparsity(report):
    for name, r in report.items():
        print(f"    {name:>6}: {r['zeros']:5d}/{r['numel']:5d} zero  ({r['sparsity'] * 100:5.1f}%)")


def export_plain_state_dict(net, layer_names):
    """Copy the net, bake the masks into the weights, return a state_dict with plain keys."""
    net_copy = copy.deepcopy(net)
    for name in layer_names:
        module = net_copy[PRUNABLE[name]]
        if prune.is_pruned(module):
            prune.remove(module, "weight")
    return net_copy.state_dict()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Iterative magnitude pruning of the ST-MNIST SCNN.")
    parser.add_argument("--model", default=os.path.join(SCRIPT_DIR, "best_scnn_net_20260817_171949_acc83.4.pth"),
                        help="Dense checkpoint to prune")
    parser.add_argument("--sparsities", type=float, nargs="+", default=[0.3, 0.5, 0.7, 0.8, 0.9],
                        help="Cumulative global sparsity targets, ascending (fraction of weights set to 0)")
    parser.add_argument("--layers", nargs="+", default=["conv1", "conv2", "fc"], choices=list(PRUNABLE),
                        help="Layers included in global pruning")
    parser.add_argument("--finetune_epochs", type=int, default=5, help="Fine-tune epochs after each pruning step")
    parser.add_argument("--lr", type=float, default=1e-3, help="Fine-tune learning rate (training used 2e-2)")
    parser.add_argument("--max_batches", type=int, default=None,
                        help="Limit train batches per epoch (quick smoke tests only)")
    parser.add_argument("--out_dir", default=os.path.join(SCRIPT_DIR, "pruned_models"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if sorted(args.sparsities) != args.sparsities or not all(0.0 < s < 1.0 for s in args.sparsities):
        parser.error("--sparsities must be ascending values in (0, 1)")

    set_seed(args.seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Device: {device}")
    os.makedirs(args.out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    trainloader, valloader, testloader = build_loaders(args.seed)

    net = build_net(device)
    net.load_state_dict(torch.load(args.model, map_location=device))
    loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)

    # ---- Baseline -----------------------------------------------------
    base_val = evaluate(net, valloader, device)
    base_test = evaluate(net, testloader, device)
    print(f"\nDense baseline | val {base_val * 100:.2f}% | test {base_test * 100:.2f}%")
    print_sparsity(layer_sparsity_report(net, args.layers))

    summary = {
        "source_model": os.path.basename(args.model),
        "layers": args.layers,
        "finetune_epochs": args.finetune_epochs,
        "lr": args.lr,
        "seed": args.seed,
        "baseline": {"val_acc": base_val, "test_acc": base_test},
        "levels": [],
    }

    prev_sparsity = 0.0
    start_time = time.time()

    for target in args.sparsities:
        print("\n" + "=" * 70)
        print(f"Target global sparsity: {target * 100:.0f}%")
        print("=" * 70)

        # global_unstructured on an already-pruned module prunes a fraction of the
        # *remaining* weights, so convert the cumulative target into a step amount.
        step_amount = (target - prev_sparsity) / (1.0 - prev_sparsity)
        prune.global_unstructured(
            prune_targets(net, args.layers),
            pruning_method=prune.L1Unstructured,
            amount=step_amount,
        )
        prev_sparsity = target

        report = layer_sparsity_report(net, args.layers)
        print_sparsity(report)

        pruned_val = evaluate(net, valloader, device)
        print(f"  After pruning, before fine-tune | val {pruned_val * 100:.2f}%")

        # Fresh optimizer per level: parameters are now weight_orig (+ mask buffers).
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999))

        best_val = pruned_val
        best_state = copy.deepcopy(net.state_dict())  # contains weight_orig + weight_mask

        for epoch in range(args.finetune_epochs):
            net.train()
            losses = []
            for i, (data, targets) in enumerate(trainloader):
                if args.max_batches is not None and i >= args.max_batches:
                    break
                data, targets = data.to(device), targets.to(device)
                spk_rec = forward_pass(net, data)
                loss_val = loss_fn(spk_rec, targets)
                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()
                losses.append(loss_val.item())

            val_acc = evaluate(net, valloader, device)
            marker = ""
            if val_acc > best_val:
                best_val = val_acc
                best_state = copy.deepcopy(net.state_dict())
                marker = "  <- best"
            print(f"  Fine-tune epoch {epoch} | loss {np.mean(losses):.4f} | val {val_acc * 100:.2f}%{marker}")

        # Continue the next level from the best fine-tuned weights of this level.
        net.load_state_dict(best_state)
        test_acc = evaluate(net, testloader, device)
        report = layer_sparsity_report(net, args.layers)

        out_name = f"pruned_scnn_s{int(round(target * 100)):02d}_val{best_val * 100:.1f}_test{test_acc * 100:.1f}.pth"
        out_path = os.path.join(args.out_dir, out_name)
        plain_sd = export_plain_state_dict(net, args.layers)
        torch.save(plain_sd, out_path)

        # Sanity check: the saved file loads into the unchanged architecture and gives the same accuracy.
        check_net = build_net(device)
        check_net.load_state_dict(torch.load(out_path, map_location=device))
        check_val = evaluate(check_net, valloader, device)
        assert abs(check_val - best_val) < 1e-9, f"Reloaded val acc {check_val} != {best_val}"

        print(f"  Best val {best_val * 100:.2f}% | test {test_acc * 100:.2f}% | saved -> {out_path}")

        summary["levels"].append({
            "target_sparsity": target,
            "file": out_name,
            "val_acc_after_prune": pruned_val,
            "val_acc": best_val,
            "test_acc": test_acc,
            "sparsity": report,
        })

    elapsed = time.time() - start_time
    summary["elapsed_s"] = elapsed

    summary_path = os.path.join(args.out_dir, f"prune_summary_{timestamp}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  dense      | val {base_val * 100:6.2f}% | test {base_test * 100:6.2f}%")
    for lvl in summary["levels"]:
        print(f"  {lvl['target_sparsity'] * 100:3.0f}% sparse | val {lvl['val_acc'] * 100:6.2f}% | "
              f"test {lvl['test_acc'] * 100:6.2f}% | {lvl['file']}")
    print(f"\nElapsed: {elapsed / 60:.1f} min. Summary written to {summary_path}")


if __name__ == "__main__":
    main()

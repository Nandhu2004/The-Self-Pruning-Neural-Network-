"""
Self-Pruning Neural Network on CIFAR-10 using learnable sigmoid gates.

The core idea:
  - Every weight in each linear layer has a corresponding learnable gate score.
  - During the forward pass, gates = sigmoid(gate_scores * temperature) in (0, 1).
  - The effective weight = raw_weight * gate.
  - An L1 penalty (lambda * mean(gates)) is added to the cross-entropy loss,
    pushing gate values toward 0, effectively pruning those weights.
  - After training, weights with gate < 0.5 are considered pruned.

Usage:
    python self_pruning_net_commented.py                 # full run (30 epochs)
    python self_pruning_net_commented.py --quick         # 3-epoch smoke test
    python self_pruning_net_commented.py --epochs 50     # custom epoch count
"""

import math
import argparse
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for scripts and servers
import matplotlib.pyplot as plt
import numpy as np


class PrunableLinear(nn.Module):
    """
    A drop-in replacement for nn.Linear with a per-weight learnable gate.

    For each weight w_ij, we introduce a learnable scalar gate_score_ij.
    During the forward pass:
        gate_ij         = sigmoid(gate_score_ij * temperature)   in (0, 1)
        effective_w_ij  = w_ij * gate_ij
        output          = input @ effective_W.T + bias

    Gradient flow:
        Because sigmoid is differentiable, gradients flow back through the gate
        into gate_scores, and also through the effective weight into the raw weight.
        Both weights and gates are jointly optimised end-to-end.

    Temperature:
        Scales the gate_scores before sigmoid, sharpening the S-curve.
        temperature = 1  -> standard sigmoid, gates cluster around 0.5
        temperature > 1  -> harder decisions, gates pushed toward 0 or 1
        temperature < 1  -> softer decisions, less pruning pressure

    Hard pruning (post-training):
        Call hard_prune(threshold) to zero-out weights whose gate is below
        the threshold, converting soft continuous gates into a binary mask.
    """

    def __init__(self, in_features: int, out_features: int, temperature: float = 2.0) -> None:
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.temperature  = temperature

        # Standard learnable weight matrix and bias, same as nn.Linear
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features))

        # One learnable gate score per weight element.
        # Initialised to 0 so sigmoid(0) = 0.5, starting with no pruning bias.
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # Kaiming uniform init for weights and uniform init for bias, matching nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute soft gates in (0, 1), then apply them element-wise to weights
        gates         = torch.sigmoid(self.gate_scores * self.temperature)
        pruned_weight = self.weight * gates
        return F.linear(x, pruned_weight, self.bias)

    def hard_prune(self, threshold: float = 0.5) -> None:
        # Convert soft gates to a binary mask and zero out pruned weights in-place.
        # After this call, the weight matrix is exactly sparse (has true zeros).
        with torch.no_grad():
            gates = torch.sigmoid(self.gate_scores * self.temperature)
            mask  = (gates >= threshold).float()
            self.weight.mul_(mask)
            self.gate_scores.requires_grad_(False)  # freeze gates post-pruning

    def gate_values(self) -> torch.Tensor:
        # Return current gate values detached on CPU, used for analysis and plotting
        with torch.no_grad():
            return torch.sigmoid(self.gate_scores * self.temperature).detach().cpu()

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"temperature={self.temperature}")


class SelfPruningNet(nn.Module):
    """
    Four-layer MLP for CIFAR-10 classification.
    Architecture: 3072 -> 1024 -> 512 -> 256 -> 10

    All linear layers use PrunableLinear so every weight has a learnable gate.
    BatchNorm and ReLU are standard with no modifications.
    """

    def __init__(self, temperature: float = 2.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            PrunableLinear(3072, 1024, temperature),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            PrunableLinear(1024, 512, temperature),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            PrunableLinear(512, 256, temperature),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            PrunableLinear(256, 10, temperature),  # output logits, no softmax needed
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)  # flatten (B, 3, 32, 32) -> (B, 3072)
        return self.net(x)

    def prunable_layers(self) -> List[PrunableLinear]:
        return [m for m in self.modules() if isinstance(m, PrunableLinear)]

    def sparsity_loss(self) -> torch.Tensor:
        """
        Returns the mean gate value across all gate parameters.

        This acts as an L1 penalty on the gates. Since gate values are in (0, 1),
        minimising the mean gate is equivalent to minimising their L1 norm.
        The optimiser is incentivised to push gate_scores negative, which maps
        gate values toward 0 via sigmoid, effectively pruning those weights.

        We normalise by the total gate count so the sparsity term stays on the
        same scale as cross-entropy regardless of model size, making lambda
        a meaningful and architecture-independent hyperparameter.
        """
        total_gates = torch.tensor(0.0, device=next(self.parameters()).device)
        total_count = 0
        for layer in self.prunable_layers():
            g            = torch.sigmoid(layer.gate_scores * layer.temperature)
            total_gates  = total_gates + g.sum()
            total_count += g.numel()
        return total_gates / total_count  # scalar in (0, 1)

    def sparsity_level(self, threshold: float = 0.5) -> float:
        # Fraction of gates below the threshold, i.e., effectively pruned weights
        all_gates = torch.cat([layer.gate_values().flatten() for layer in self.prunable_layers()])
        return (all_gates < threshold).float().mean().item()

    def all_gate_values(self) -> np.ndarray:
        # Concatenate all gate values across all layers into one flat numpy array
        return torch.cat([
            layer.gate_values().flatten() for layer in self.prunable_layers()
        ]).numpy()

    def compression_report(self, threshold: float = 0.5) -> dict:
        """
        Counts active vs pruned weights and computes compression ratio.
        compression_ratio = total / active, e.g. 4x means 75% of weights are pruned.
        """
        total = 0
        active = 0
        for layer in self.prunable_layers():
            gates  = layer.gate_values()
            total  += gates.numel()
            active += (gates >= threshold).sum().item()
        pruned = total - active
        ratio  = total / active if active > 0 else float("inf")
        return {
            "total_params":      total,
            "effective_params":  active,
            "pruned_params":     pruned,
            "compression_ratio": ratio,
        }

    def apply_hard_pruning(self, threshold: float = 0.5) -> None:
        for layer in self.prunable_layers():
            layer.hard_prune(threshold)


def get_cifar10_loaders(batch_size: int = 128) -> Tuple[DataLoader, DataLoader]:
    """
    Load CIFAR-10 with standard normalisation.

    Training uses RandomCrop and RandomHorizontalFlip for regularisation.
    Test set uses only normalisation for unbiased evaluation.
    """
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = torchvision.datasets.CIFAR10(root="./data", train=True,  download=True, transform=train_tf)
    test_ds  = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tf)

    # num_workers=0 loads data on the main process, avoiding multiprocessing
    # issues on Windows. pin_memory is disabled since there is no GPU.
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=256,        shuffle=False, num_workers=0, pin_memory=False)

    return train_loader, test_loader


def train_one_epoch(
    model:     SelfPruningNet,
    loader:    DataLoader,
    optimiser: torch.optim.Optimizer,
    lam:       float,
    device:    torch.device,
) -> Tuple[float, float]:
    """
    Run one full training epoch.

    Total loss = CrossEntropy(logits, labels) + lambda * mean_gate_value

    The sparsity term penalises gates being open (close to 1). The optimiser
    therefore pushes gate_scores negative, moving gate values toward 0 via
    sigmoid, which prunes those weights. Lambda controls the trade-off:
    higher lambda = more pruning, potentially lower accuracy.
    """
    model.train()
    total_cls = total_sparse = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        logits      = model(images)
        cls_loss    = F.cross_entropy(logits, labels)
        sparse_loss = model.sparsity_loss()          # mean gate value in (0, 1)
        total_loss  = cls_loss + lam * sparse_loss   # combined objective

        optimiser.zero_grad()
        total_loss.backward()  # gradients flow through gates into gate_scores
        optimiser.step()

        total_cls    += cls_loss.item()
        total_sparse += sparse_loss.item()

    n = len(loader)
    return total_cls / n, total_sparse / n


def evaluate(model: SelfPruningNet, loader: DataLoader, device: torch.device) -> float:
    # Compute top-1 test accuracy with no gradient computation
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds   = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return correct / total


def run_experiment(
    lam:          float,
    temperature:  float,
    train_loader: DataLoader,
    test_loader:  DataLoader,
    device:       torch.device,
    epochs:       int   = 30,
    lr:           float = 1e-3,
) -> dict:
    """
    Train a fresh model with the given lambda and temperature, then hard-prune
    and re-evaluate. Returns a results dict for reporting and plotting.

    Optimiser: Adam with weight_decay=1e-4 (L2 on weights, not on gates).
    Scheduler: CosineAnnealingLR smoothly decays LR to near zero, which
               reduces oscillation late in training and improves final accuracy.
    """
    model     = SelfPruningNet(temperature=temperature).to(device)

    # Separate learning rates: weights learn fast, gate scores learn slowly.
    # If gate scores move too fast they collapse to 0 immediately (full pruning).
    # If they move too slow they never prune. 1e-2 for weights, 1e-3 for gates works well.
    weight_params = [p for n, p in model.named_parameters() if "gate_scores" not in n]
    gate_params   = [p for n, p in model.named_parameters() if "gate_scores"     in n]
    optimiser = torch.optim.Adam([
        {"params": weight_params, "lr": lr},
        {"params": gate_params,   "lr": lr * 0.1},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    print(f"\nlambda = {lam:.0e}  |  temperature = {temperature}  |  {epochs} epochs")

    for epoch in range(1, epochs + 1):
        cls_l, spa_l = train_one_epoch(model, train_loader, optimiser, lam, device)
        scheduler.step()

        # Log every 5 epochs and on epoch 1 to track convergence
        if epoch % 2 == 0 or epoch == 1:
            acc      = evaluate(model, test_loader, device)
            sparsity = model.sparsity_level()
            print(f"  Epoch {epoch:3d} | cls={cls_l:.4f} | sparse={spa_l:.4f} "
                  f"| acc={acc*100:.2f}% | sparsity={sparsity*100:.1f}%")

    # Evaluate with soft (continuous) gates still active
    soft_acc      = evaluate(model, test_loader, device)
    soft_sparsity = model.sparsity_level()
    gate_vals     = model.all_gate_values()   # capture before hard pruning modifies weights
    comp_report   = model.compression_report()

    # Apply hard pruning then re-evaluate to simulate real deployment.
    # A small gap between soft and hard accuracy means gates are already near-binary,
    # which is a sign of successful training with the temperature sharpening.
    model.apply_hard_pruning(threshold=0.5)
    hard_acc = evaluate(model, test_loader, device)

    print(f"  Soft accuracy : {soft_acc*100:.2f}%  (continuous gates)")
    print(f"  Hard accuracy : {hard_acc*100:.2f}%  (binary mask, deployment)")
    print(f"  Sparsity      : {soft_sparsity*100:.2f}%")
    print(f"  Total params  : {comp_report['total_params']:,}")
    print(f"  Active params : {comp_report['effective_params']:,}")
    print(f"  Compression   : {comp_report['compression_ratio']:.2f}x")

    return {
        "lam":         lam,
        "temperature": temperature,
        "soft_acc":    soft_acc,
        "hard_acc":    hard_acc,
        "sparsity":    soft_sparsity,
        "compression": comp_report["compression_ratio"],
        "gate_vals":   gate_vals,
    }


def plot_gate_distribution(
    gate_vals: np.ndarray,
    lam:       float,
    temp:      float,
    save_path: str = "gate_distribution.png",
) -> None:
    """
    Histogram of final gate values for the best model.

    A successful result shows a bimodal distribution:
      spike near 0  -> pruned weights (gate pushed to zero by L1 penalty)
      spike near 1  -> active weights (gate stayed open because they were useful)
    With temperature > 1, this bimodal shape is more pronounced because the
    sharpened sigmoid encourages harder, more decisive gate values.
    """
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(gate_vals, bins=100, color="#2E4057", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Gate Value", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(
        f"Gate Value Distribution  (lambda={lam:.0e}, temperature={temp})\n"
        "Ideal: bimodal — spike near 0 (pruned) and spike near 1 (active)",
        fontsize=11,
    )
    ax.axvline(x=0.5, color="#E84855", linestyle="--", linewidth=1.4,
               label="Hard-prune threshold (0.5)")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Gate distribution plot saved -> {save_path}")


def plot_temperature_comparison(results: list, save_path: str = "temperature_comparison.png") -> None:
    # Bar chart comparing soft accuracy, hard accuracy, and sparsity
    # across all (lambda, temperature) combinations
    labels     = [f"λ={r['lam']:.0e}\nT={r['temperature']}" for r in results]
    soft_accs  = [r["soft_acc"]  * 100 for r in results]
    hard_accs  = [r["hard_acc"]  * 100 for r in results]
    sparsities = [r["sparsity"]  * 100 for r in results]

    x   = np.arange(len(labels))
    w   = 0.28
    fig, ax1 = plt.subplots(figsize=(max(10, len(labels) * 1.4), 5))

    b1 = ax1.bar(x - w, soft_accs,  w, label="Soft Acc (%)", color="#4C72B0")
    b2 = ax1.bar(x,     hard_accs,  w, label="Hard Acc (%)", color="#55A868")
    ax1.set_ylabel("Accuracy (%)", fontsize=11)
    ax1.set_ylim(0, 100)

    ax2 = ax1.twinx()
    b3  = ax2.bar(x + w, sparsities, w, label="Sparsity (%)", color="#C44E52", alpha=0.8)
    ax2.set_ylabel("Sparsity (%)", fontsize=11)
    ax2.set_ylim(0, 100)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_title("Accuracy & Sparsity across lambda x Temperature Grid", fontsize=12)
    ax1.legend(handles=[b1, b2, b3], loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Temperature comparison plot saved -> {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-Pruning Neural Network on CIFAR-10")
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--quick",      action="store_true",
                        help="Run a 3-epoch smoke test across all lambda/temperature combos")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, test_loader = get_cifar10_loaders(args.batch_size)

    # Reduced grid: 3 lambda values, 1 fixed temperature, 10 epochs each.
    # This runs 3 experiments total and finishes in ~10 minutes on a CPU laptop.
    # To run the full grid (9 experiments x 30 epochs), change temperatures back
    # to [1.0, 2.0, 5.0] and epochs back to 30.
    lambdas      = [1e-4, 1e-3, 1e-2]
    temperatures = [5.0]
    epochs       = 3 if args.quick else min(args.epochs, 10)

    all_results = []
    best        = {"soft_acc": -1.0}

    for lam in lambdas:
        for temp in temperatures:
            res = run_experiment(lam, temp, train_loader, test_loader, device, epochs=epochs, lr=args.lr)
            all_results.append(res)
            if res["soft_acc"] > best["soft_acc"]:
                best = res

    # Print summary table
    print("\nResults Summary")
    header = f"  {'Lambda':<9} {'Temp':>6} {'Soft Acc':>10} {'Hard Acc':>10} {'Sparsity':>10} {'Compress':>10}"
    print(header)
    for r in all_results:
        print(f"  {r['lam']:<9.0e} {r['temperature']:>6.1f} "
              f"{r['soft_acc']*100:>9.2f}% "
              f"{r['hard_acc']*100:>9.2f}% "
              f"{r['sparsity']*100:>9.2f}% "
              f"{r['compression']:>9.2f}x")

    print(f"\nBest: lambda={best['lam']:.0e}, T={best['temperature']} | "
          f"Soft Acc={best['soft_acc']*100:.2f}% | "
          f"Hard Acc={best['hard_acc']*100:.2f}% | "
          f"Sparsity={best['sparsity']*100:.2f}% | "
          f"Compression={best['compression']:.2f}x")

    # Save plots for the best model and the full grid comparison
    plot_gate_distribution(best["gate_vals"], best["lam"], best["temperature"],
                           save_path="gate_distribution_best.png")
    plot_temperature_comparison(all_results, save_path="temperature_comparison.png")


if __name__ == "__main__":
    main()

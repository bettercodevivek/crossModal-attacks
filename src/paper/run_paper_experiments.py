"""
Paper-aligned experiments (Section V–VI): MNIST/CIFAR-10 CNNs, FGSM/PGD/CW,
ASR and robustness R, adversarial training, distillation, summary tables & plots.

Run from project root:
  python src/paper/run_paper_experiments.py train --dataset mnist --defense none
  python src/paper/run_paper_experiments.py eval --dataset mnist --checkpoint paper_checkpoints/mnist_none.pt --attack pgd
  python src/paper/run_paper_experiments.py reproduce-table --defense none
"""
import argparse
import os
import random
import sys

import numpy as np
import torch

# Allow `from paper.X` when launching this file directly
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from paper.data import get_loaders
from paper.metrics_classify import accuracy, accuracy_max_batches, evaluate_robustness
from paper.models import build_model
from paper.paper_config import CNN_PAPER_BENCHMARK_DIR, PaperConfig
from paper.plots import plot_asr_bars, plot_radar_fgsm_pgd, save_results_json
from paper.train_models import load_checkpoint, save_checkpoint, train_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def checkpoint_path(dataset: str, defense: str) -> str:
    return os.path.join(
        PaperConfig.checkpoint_dir(), f"{dataset.lower()}_{defense.lower()}.pt"
    )


def cmd_train(args):
    PaperConfig.ensure_dirs()
    set_seed(PaperConfig.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = args.dataset.lower()
    defense = args.defense.lower()
    epochs = args.epochs or (
        PaperConfig.EPOCHS_MNIST if dataset == "mnist" else PaperConfig.EPOCHS_CIFAR
    )

    train_loader, test_loader = get_loaders(
        dataset, PaperConfig.BATCH_SIZE, num_workers=args.num_workers
    )

    if defense == "distillation":
        if not args.teacher_ckpt:
            raise SystemExit("--teacher_ckpt required for distillation defense")
        teacher = build_model(dataset, device)
        load_checkpoint(teacher, args.teacher_ckpt, device)
        student = build_model(dataset, device)
        train_model(
            student,
            train_loader,
            test_loader,
            device,
            epochs,
            PaperConfig.LR,
            PaperConfig.WEIGHT_DECAY,
            defense="distillation",
            teacher=teacher,
            cfg=PaperConfig,
        )
        out = args.output or checkpoint_path(dataset, "distill")
        save_checkpoint(student, out)
        print(f"Saved student to {out}")
        return

    model = build_model(dataset, device)
    train_model(
        model,
        train_loader,
        test_loader,
        device,
        epochs,
        PaperConfig.LR,
        PaperConfig.WEIGHT_DECAY,
        defense=defense if defense != "none" else "none",
        teacher=None,
        cfg=PaperConfig,
    )
    out = args.output or checkpoint_path(dataset, defense)
    save_checkpoint(model, out)
    print(f"Saved checkpoint to {out}")


def cmd_eval(args):
    PaperConfig.ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = args.dataset.lower()
    model = build_model(dataset, device)
    load_checkpoint(model, args.checkpoint, device)

    _, test_loader = get_loaders(
        dataset, PaperConfig.BATCH_SIZE, num_workers=args.num_workers
    )
    clean_acc = accuracy(model, test_loader, device)
    print(f"Clean test accuracy: {clean_acc:.4f}")

    metrics = evaluate_robustness(
        model,
        test_loader,
        device,
        attack=args.attack.lower(),
        epsilon=args.epsilon,
        pgd_alpha=args.pgd_alpha,
        pgd_steps=args.pgd_steps,
        cw_steps=args.cw_steps,
        cw_lr=args.cw_lr,
        cw_c=args.cw_c,
        max_batches=args.limit_batches,
    )
    print(f"ASR (misclassified after attack / evaluated): {metrics['asr_percent']:.2f}%")
    print(f"Robustness R = 1 - ASR: {metrics['robustness_score']:.4f}")

    row = {
        "dataset": dataset,
        "attack": args.attack.lower(),
        "asr_percent": metrics["asr_percent"],
        "robustness_score": metrics["robustness_score"],
        "clean_accuracy": clean_acc,
    }
    if args.output_json:
        save_results_json([row], args.output_json)
        print(f"Wrote {args.output_json}")


def cmd_reproduce_table(args):
    """Train (optional) and evaluate FGSM & PGD on MNIST and CIFAR-10; save Table 2–style results."""
    PaperConfig.ensure_dirs()
    set_seed(PaperConfig.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    defense = args.defense.lower()
    if args.epochs is not None:
        epochs_mnist = epochs_cifar = args.epochs
    elif args.quick:
        epochs_mnist = min(3, PaperConfig.EPOCHS_MNIST)
        epochs_cifar = min(5, PaperConfig.EPOCHS_CIFAR)
    else:
        epochs_mnist = PaperConfig.EPOCHS_MNIST
        epochs_cifar = PaperConfig.EPOCHS_CIFAR

    results = []
    for dataset in ["mnist", "cifar10"]:
        ckpt = checkpoint_path(dataset, defense)
        if not args.skip_train or not os.path.isfile(ckpt):
            print(f"\n=== Training {dataset} ({defense}) ===")
            ep = epochs_mnist if dataset == "mnist" else epochs_cifar
            if args.quick:
                ep = epochs_mnist if dataset == "mnist" else epochs_cifar
            train_loader, test_loader = get_loaders(
                dataset, PaperConfig.BATCH_SIZE, num_workers=args.num_workers
            )
            model = build_model(dataset, device)
            train_model(
                model,
                train_loader,
                test_loader,
                device,
                ep,
                PaperConfig.LR,
                PaperConfig.WEIGHT_DECAY,
                defense=defense if defense != "none" else "none",
                teacher=None,
                cfg=PaperConfig,
            )
            save_checkpoint(model, ckpt)
            print(f"Saved {ckpt}")
        else:
            print(f"Using existing checkpoint {ckpt}")

        model = build_model(dataset, device)
        load_checkpoint(model, ckpt, device)
        _, test_loader = get_loaders(
            dataset, PaperConfig.BATCH_SIZE, num_workers=args.num_workers
        )

        for attack_name in ["fgsm", "pgd"]:
            print(f"\n=== {dataset} / {attack_name.upper()} ===")
            eps = (
                PaperConfig.FGSM_EPSILON
                if attack_name == "fgsm"
                else PaperConfig.PGD_EPSILON
            )
            m = evaluate_robustness(
                model,
                test_loader,
                device,
                attack=attack_name,
                epsilon=eps,
                pgd_alpha=PaperConfig.PGD_ALPHA,
                pgd_steps=PaperConfig.PGD_STEPS,
                cw_steps=PaperConfig.CW_STEPS,
                cw_lr=PaperConfig.CW_LR,
                cw_c=PaperConfig.CW_C,
                max_batches=args.limit_batches,
            )
            results.append(
                {
                    "dataset": dataset,
                    "attack": attack_name,
                    "asr_percent": m["asr_percent"],
                    "robustness_score": m["robustness_score"],
                }
            )
            print(f"ASR: {m['asr_percent']:.2f}%  R: {m['robustness_score']:.4f}")

    out_dir = PaperConfig.results_dir()
    json_path = os.path.join(out_dir, "table2_asr.json")
    save_results_json(results, json_path)
    print(f"\nSaved summary to {json_path}")

    bar_path = os.path.join(out_dir, "asr_bars.png")
    plot_asr_bars(results, bar_path, title="Attack Success Rate (%) — FGSM vs PGD")
    print(f"Saved plot {bar_path}")

    radar_path = os.path.join(out_dir, "asr_radar_fgsm_pgd.png")
    plot_radar_fgsm_pgd(results, radar_path)
    print(f"Saved plot {radar_path}")


def cmd_fast_benchmark(args):
    """
    Subset training + reduced PGD depth + capped eval batches.
    Writes only JSON + PNGs under cnn_paper_benchmark/ (safe to commit).
    Checkpoints stay under paper_checkpoints/ (gitignored).
    """
    os.makedirs(CNN_PAPER_BENCHMARK_DIR, exist_ok=True)
    PaperConfig.ensure_dirs()
    set_seed(PaperConfig.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    defense = "none"
    bs = PaperConfig.FAST_BATCH_SIZE
    epochs_m = PaperConfig.FAST_TRAIN_EPOCHS
    epochs_c = PaperConfig.FAST_TRAIN_EPOCHS_CIFAR
    lim = args.limit_batches or PaperConfig.FAST_EVAL_MAX_BATCHES
    pgd_fast = args.pgd_steps or PaperConfig.FAST_PGD_STEPS

    meta = {
        "benchmark": "cnn_fast",
        "note": "Quick CNN benchmark for papers/CI; subset training and capped eval batches — not full-test ASR.",
        "train_epochs_mnist": epochs_m,
        "train_epochs_cifar10": epochs_c,
        "train_subset_mnist": PaperConfig.FAST_TRAIN_SUBSET_MNIST,
        "train_subset_cifar10": PaperConfig.FAST_TRAIN_SUBSET_CIFAR,
        "eval_max_batches": lim,
        "batch_size": bs,
        "pgd_steps_eval": pgd_fast,
        "epsilon_fgsm_pgd": PaperConfig.FGSM_EPSILON,
        "device": str(device),
        "clean_accuracy_eval_subset": {},
    }

    results = []
    for dataset in ["mnist", "cifar10"]:
        ckpt = checkpoint_path(dataset, defense)
        subset_n = (
            PaperConfig.FAST_TRAIN_SUBSET_MNIST
            if dataset == "mnist"
            else PaperConfig.FAST_TRAIN_SUBSET_CIFAR
        )
        ep = epochs_m if dataset == "mnist" else epochs_c

        if not args.skip_train or not os.path.isfile(ckpt):
            ep_label = "epoch" if ep == 1 else "epochs"
            print(
                f"\n=== [fast] Training {dataset} ({subset_n} samples, {ep} {ep_label}) ==="
            )
            train_loader, test_loader = get_loaders(
                dataset,
                bs,
                num_workers=args.num_workers,
                train_max_samples=subset_n,
            )
            model = build_model(dataset, device)
            train_model(
                model,
                train_loader,
                test_loader,
                device,
                ep,
                PaperConfig.LR,
                PaperConfig.WEIGHT_DECAY,
                defense="none",
                teacher=None,
                cfg=PaperConfig,
            )
            save_checkpoint(model, ckpt)
            print(f"Saved {ckpt}")
        else:
            print(f"Using existing checkpoint {ckpt}")

        model = build_model(dataset, device)
        load_checkpoint(model, ckpt, device)
        _, test_loader = get_loaders(dataset, bs, num_workers=args.num_workers)
        clean_sub = accuracy_max_batches(model, test_loader, device, lim)
        meta["clean_accuracy_eval_subset"][dataset] = round(float(clean_sub), 4)
        print(
            f"[fast] Clean accuracy (first {lim} test batches, up to {lim * bs} images): {clean_sub:.4f}"
        )

        for attack_name in ["fgsm", "pgd"]:
            print(f"\n=== [fast] {dataset} / {attack_name.upper()} ===")
            eps = (
                PaperConfig.FGSM_EPSILON
                if attack_name == "fgsm"
                else PaperConfig.PGD_EPSILON
            )
            m = evaluate_robustness(
                model,
                test_loader,
                device,
                attack=attack_name,
                epsilon=eps,
                pgd_alpha=PaperConfig.PGD_ALPHA,
                pgd_steps=pgd_fast if attack_name == "pgd" else PaperConfig.PGD_STEPS,
                cw_steps=PaperConfig.CW_STEPS,
                cw_lr=PaperConfig.CW_LR,
                cw_c=PaperConfig.CW_C,
                max_batches=lim,
            )
            row = {
                "dataset": dataset,
                "attack": attack_name,
                "asr_percent": round(m["asr_percent"], 2),
                "robustness_score": round(m["robustness_score"], 4),
                "evaluated_samples": m["total_samples"],
            }
            results.append(row)
            print(f"ASR: {row['asr_percent']:.2f}%  R: {row['robustness_score']:.4f}")

    out_payload = {"meta": meta, "results": results}
    json_path = os.path.join(CNN_PAPER_BENCHMARK_DIR, "table2_asr.json")
    save_results_json(out_payload, json_path)
    print(f"\nSaved {json_path}")

    bar_path = os.path.join(CNN_PAPER_BENCHMARK_DIR, "asr_bars.png")
    plot_asr_bars(
        results,
        bar_path,
        title="Attack Success Rate (%) — FGSM vs PGD (fast benchmark)",
    )
    print(f"Saved plot {bar_path}")

    radar_path = os.path.join(CNN_PAPER_BENCHMARK_DIR, "asr_radar_fgsm_pgd.png")
    plot_radar_fgsm_pgd(results, radar_path)
    print(f"Saved plot {radar_path}")


def main():
    p = argparse.ArgumentParser(
        description="Paper reproduction: MNIST/CIFAR CNN robustness (FGSM, PGD, CW)"
    )
    sub = p.add_subparsers(dest="command", required=True)

    pt = sub.add_parser("train", help="Train CNN with optional defense")
    pt.add_argument("--dataset", choices=["mnist", "cifar10"], required=True)
    pt.add_argument(
        "--defense",
        choices=["none", "adversarial", "distillation"],
        default="none",
    )
    pt.add_argument("--epochs", type=int, default=None)
    pt.add_argument("--output", type=str, default=None)
    pt.add_argument("--teacher_ckpt", type=str, default=None)
    pt.add_argument("--num_workers", type=int, default=0)
    pt.set_defaults(func=cmd_train)

    pe = sub.add_parser("eval", help="Evaluate ASR / robustness on a checkpoint")
    pe.add_argument("--dataset", choices=["mnist", "cifar10"], required=True)
    pe.add_argument("--checkpoint", type=str, required=True)
    pe.add_argument("--attack", choices=["fgsm", "pgd", "cw"], required=True)
    pe.add_argument("--epsilon", type=float, default=PaperConfig.FGSM_EPSILON)
    pe.add_argument("--pgd_alpha", type=float, default=PaperConfig.PGD_ALPHA)
    pe.add_argument("--pgd_steps", type=int, default=PaperConfig.PGD_STEPS)
    pe.add_argument("--cw_steps", type=int, default=PaperConfig.CW_STEPS)
    pe.add_argument("--cw_lr", type=float, default=PaperConfig.CW_LR)
    pe.add_argument("--cw_c", type=float, default=PaperConfig.CW_C)
    pe.add_argument("--limit_batches", type=int, default=None)
    pe.add_argument("--output_json", type=str, default=None)
    pe.add_argument("--num_workers", type=int, default=0)
    pe.set_defaults(func=cmd_eval)

    pr = sub.add_parser(
        "reproduce-table",
        help="Train MNIST+CIFAR (unless --skip_train) and FGSM/PGD ASR table + plots",
    )
    pr.add_argument(
        "--defense", choices=["none", "adversarial"], default="none"
    )
    pr.add_argument("--skip_train", action="store_true")
    pr.add_argument("--quick", action="store_true", help="Fewer epochs for smoke test")
    pr.add_argument("--epochs", type=int, default=None)
    pr.add_argument("--limit_batches", type=int, default=None)
    pr.add_argument("--num_workers", type=int, default=0)
    pr.set_defaults(func=cmd_reproduce_table)

    pf = sub.add_parser(
        "fast-benchmark",
        help="~2–3 min CNN benchmark; writes cnn_paper_benchmark/* (JSON+PNG, git-friendly)",
    )
    pf.add_argument(
        "--skip_train",
        action="store_true",
        help="Use existing paper_checkpoints/mnist_none.pt and cifar10_none.pt",
    )
    pf.add_argument(
        "--limit_batches",
        type=int,
        default=None,
        help="Cap test batches for attack eval (default from PaperConfig.FAST_EVAL_MAX_BATCHES)",
    )
    pf.add_argument(
        "--pgd_steps",
        type=int,
        default=None,
        help="PGD steps during eval only (default FAST_PGD_STEPS)",
    )
    pf.add_argument("--num_workers", type=int, default=0)
    pf.set_defaults(func=cmd_fast_benchmark)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

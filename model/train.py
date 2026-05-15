import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from coolz.logger import get_logger
from coolz.dataset import CLASS_NAMES, build_df, make_folds, EEGDataset
from coolz.model import EEGNet

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / 'data'
CKPT_ROOT = ROOT_DIR / 'checkpoints'
LOGS_ROOT = ROOT_DIR / 'logs'


def _next_version() -> int:
    existing = [d for d in CKPT_ROOT.glob('coolz_v*') if d.is_dir()]
    if not existing:
        return 1
    return max(int(d.name[7:]) for d in existing) + 1


def kldiv_loss(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    per = F.kl_div(log_probs, targets, reduction='none').sum(dim=1)  # (B,)
    if weights is None:
        return per.mean()
    return (per * weights).mean()


def run_epoch(
    model, loader, optimizer, device,
    train: bool, desc: str = '', use_weights: bool = False,
):
    model.train(train)
    total_loss, n = 0.0, 0
    all_probs, all_targets = [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        bar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
        for signals, labels, weights in bar:
            signals = signals.to(device)
            labels = labels.to(device)
            w = weights.to(device) if use_weights else None
            log_probs = model(signals)
            loss = kldiv_loss(log_probs, labels, w)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                all_probs.append(log_probs.exp().cpu())
                all_targets.append(labels.cpu())
            total_loss += loss.item() * len(signals)
            n += len(signals)
            bar.set_postfix(loss=f'{total_loss / n:.4f}')
    if train:
        return total_loss / n, None, None
    return total_loss / n, torch.cat(all_probs), torch.cat(all_targets)


def _summarise(preds: torch.Tensor) -> str:
    m = preds.mean(dim=0)
    return '  '.join(f'{c}={m[i]:.3f}' for i, c in enumerate(CLASS_NAMES))


def train_fold(args, df, fold: int, device, ckpt_dir: Path, log_dir: Path, log) -> float:
    train_df = df[df.fold != fold]
    val_df = df[df.fold == fold]
    eeg_dir = DATA_DIR / 'train_eegs'

    val_ds = EEGDataset(val_df, eeg_dir, augment=False, min_votes=args.val_min_votes)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    model = EEGNet(args.backbone, args.dropout).to(device)

    metrics_path = log_dir / 'metrics.csv'
    mf = metrics_path.open('a', newline='')
    mw = csv.DictWriter(mf, fieldnames=['fold', 'stage', 'epoch', 'train_loss', 'val_loss'])
    if not metrics_path.exists() or metrics_path.stat().st_size == 0:
        mw.writeheader()

    best_val = float('inf')
    best_ckpt: Path | None = None

    def _run_stage(stage: int, epochs: int, lr: float, min_votes: int, use_weights: bool):
        nonlocal best_val, best_ckpt

        tr_ds = EEGDataset(train_df, eeg_dir, augment=True, min_votes=min_votes)
        tr_loader = DataLoader(
            tr_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True,
        )

        optimizer = torch.optim.AdamW(
            [
                {'params': model.net.parameters(), 'lr': lr / 10},
                {'params': list(model.pool.parameters()) +
                 list(model.drop.parameters()) +
                 list(model.fc.parameters()), 'lr': lr},
            ],
            weight_decay=1e-2,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        patience_count = 0

        log.info(f'fold {fold} | stage {stage} | train={len(tr_ds)} val={len(val_ds)}')

        for epoch in range(1, epochs + 1):
            train_loss, _, _ = run_epoch(
                model, tr_loader, optimizer, device, train=True,
                desc=f'f{fold} s{stage} e{epoch:03d} train', use_weights=use_weights,
            )
            val_loss, preds, _ = run_epoch(
                model, val_loader, optimizer, device, train=False,
                desc=f'f{fold} s{stage} e{epoch:03d} val  ',
            )
            scheduler.step()

            mw.writerow({
                'fold': fold, 'stage': stage, 'epoch': epoch,
                'train_loss': round(train_loss, 6), 'val_loss': round(val_loss, 6),
            })
            mf.flush()

            improved = val_loss < best_val
            if improved:
                if best_ckpt and best_ckpt.exists():
                    best_ckpt.unlink()
                best_val = val_loss
                best_ckpt = ckpt_dir / f'fold{fold}_s{stage}_e{epoch:03d}_val{val_loss:.4f}.ckpt'
                torch.save({
                    'backbone': args.backbone,
                    'dropout': args.dropout,
                    'state_dict': model.state_dict(),
                }, best_ckpt)
                patience_count = 0
            else:
                patience_count += 1

            log.info(
                f'fold {fold} | s{stage} epoch {epoch:03d}/{epochs} '
                f'train={train_loss:.4f} val={val_loss:.4f} '
                f'best={best_val:.4f} {"★" if improved else ""}'
            )
            log.info(f'         pred:  {_summarise(preds)}')

            if patience_count >= args.patience:
                log.info(f'fold {fold} | s{stage} early stop at epoch {epoch}')
                break

    # Stage 1: all data, loss weighted by n_votes/20
    _run_stage(1, args.s1_epochs, args.s1_lr, min_votes=0, use_weights=True)

    # Restore best Stage-1 weights — in-memory model may be at a worse last epoch
    if best_ckpt and best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location=device)['state_dict'])
        log.info(f'fold {fold} | restored best s1 weights → {best_ckpt.name}')

    # Stage 2: high-quality annotations only, uniform loss, lower LR
    _run_stage(2, args.s2_epochs, args.s2_lr, min_votes=args.s2_min_votes, use_weights=False)

    mf.close()
    log.info(f'fold {fold} | best val={best_val:.4f} → {best_ckpt.name}')
    return best_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', default='efficientnet_b5')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--s1_epochs', type=int, default=15)
    parser.add_argument('--s1_lr', type=float, default=1e-3)
    parser.add_argument('--s2_epochs', type=int, default=5)
    parser.add_argument('--s2_lr', type=float, default=1e-4)
    parser.add_argument('--s2_min_votes', type=int, default=10)
    parser.add_argument('--val_min_votes', type=int, default=10)
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--fold', type=int, default=None,
                        help='train single fold; omit for all folds')
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    version = f'coolz_v{_next_version()}'
    ckpt_dir = CKPT_ROOT / version
    log_dir = LOGS_ROOT / version
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    log = get_logger(__name__, str(log_dir))
    log.info(f'version={version}  backbone={args.backbone}')

    device = (
        torch.device('cuda') if torch.cuda.is_available() else
        torch.device('mps') if torch.backends.mps.is_available() else
        torch.device('cpu')
    )
    log.info(f'device={device}')

    df = build_df(DATA_DIR / 'train.csv')
    df = make_folds(df, n_splits=args.n_folds, seed=args.seed)

    folds = [args.fold] if args.fold is not None else list(range(args.n_folds))
    scores = []
    for fold in folds:
        score = train_fold(args, df, fold, device, ckpt_dir, log_dir, log)
        scores.append(score)

    log.info(f'CV: {[round(s, 4) for s in scores]}  mean={np.mean(scores):.4f}')


if __name__ == '__main__':
    main()

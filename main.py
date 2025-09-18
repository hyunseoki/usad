import argparse, os, datetime
import lightning.pytorch as pl
import torch
import numpy as np
import load_data
from data_paths import NORMAL_DATA_PATHS, LEAK_DATA_PATHS
from usad import UsadModel, training, testing, plot_history
from utils import to_device, ROC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def get_train_dataset(args):
    normal_data = []

    for idx, site in enumerate(args.train_site):
        signals = load_data.load_data(NORMAL_DATA_PATHS[site])
        normal_data.extend(signals)

    normal_data = np.array(normal_data, dtype=np.float32)
    return normal_data

def get_val_dataset(args):
    normal_data = load_data.load_data(NORMAL_DATA_PATHS[args.val_site])
    leak_data = load_data.load_data(LEAK_DATA_PATHS[args.val_site])

    data = np.concatenate([normal_data, leak_data], axis=0, dtype=np.float32)
    labels = np.array([0] * len(normal_data) + [1] * len(leak_data), dtype=int)

    return data, labels


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--n_epochs', type=int, default=100)

    parser.add_argument('--train_site', nargs='+', default=['a', 'b'])
    parser.add_argument('--val_site', type=str, choices=['a', 'b', 'c'], default='c')
    parser.add_argument('--log_dir', type=str, default='./checkpoint')
    parser.add_argument('--comments', type=str, default='USAD_baseline')

    return parser.parse_args()


def main():
    pl.seed_everything(42)
    args = parse_args()
    args.log_dir = os.path.join(args.log_dir, args.comments, datetime.datetime.now().strftime("%m%d%H%M%S"))
    os.makedirs(args.log_dir, exist_ok=True)
    with open(os.path.join(args.log_dir, 'config.txt'), 'w') as f:
        for key, value in vars(args).items():
            f.write('{} : {}\n'.format(key, value)) 

    train_data = get_train_dataset(args)
    train_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(train_data)
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=0
    )

    w_size = 320
    z_size = w_size * 100

    model = UsadModel(w_size, z_size)
    model = to_device(model, f'cuda:{args.device}')
    history = training(args.n_epochs, model, train_loader, train_loader)
    plot_history(history, save_fn=os.path.join(args.log_dir, 'loss.png'))

    torch.save({
                'encoder': model.encoder.state_dict(),
                'decoder1': model.decoder1.state_dict(),
                'decoder2': model.decoder2.state_dict()
                }, os.path.join(args.log_dir, "model.pth"))

    # ===== Validation & Metric 계산 =====
    val_data, y_true = get_val_dataset(args)
    val_datset = torch.utils.data.TensorDataset(
        torch.from_numpy(val_data)
    )
    val_loader = torch.utils.data.DataLoader(
        val_datset,
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=0
    )
    results = testing(model, val_loader)
    y_pred = np.concatenate([
        torch.stack(results[:-1]).flatten().detach().cpu().numpy(),
        results[-1].flatten().detach().cpu().numpy()
    ])

    # ===== Threshold 탐색 =====
    thresholds = np.linspace(y_pred.min(), y_pred.max(), 500)
    best_f1, best_th = -1, 0
    for th in thresholds:
        y_hat = (y_pred > th).astype(int)
        f1 = f1_score(y_true, y_hat)
        if f1 > best_f1:
            best_f1, best_th = f1, th

    # ===== 최종 Metric 계산 =====
    y_hat = (y_pred > best_th).astype(int)
    acc  = accuracy_score(y_true, y_hat)
    prec = precision_score(y_true, y_hat)
    rec  = recall_score(y_true, y_hat)
    f1   = f1_score(y_true, y_hat)
    auc  = roc_auc_score(y_true, y_pred)
    if auc < 0.5:
        auc = 1 - auc

    # ===== 파일 저장 =====
    metric_path = os.path.join(args.log_dir, "metric.txt")
    with open(metric_path, "w") as f:
        f.write(f"Best threshold : {best_th:.3f}\n")
        f.write(f"Accuracy       : {acc:.3f}\n")
        f.write(f"Precision      : {prec:.3f}\n")
        f.write(f"Recall         : {rec:.3f}\n")
        f.write(f"F1 Score       : {f1:.3f}\n")
        f.write(f"AUC-ROC        : {auc:.3f}\n")

    print(f"Metrics saved to {metric_path}")
    # threshold = ROC(y_true,y_pred, save_fn=os.path.join(args.log_dir, 'rocauc.png'))

if __name__ == '__main__':
    main()
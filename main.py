from usad import Encoder, Decoder
import argparse, os, datetime
import lightning.pytorch as pl
import torch
import numpy as np
import load_data
from data_paths import NORMAL_DATA_PATHS, LEAK_DATA_PATHS
from usad import UsadModel, training
from utils import to_device


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
    
    # --- USAD 관련 하이퍼파라미터 추가 ---
    parser.add_argument('--input_size', type=int, default=320, help='Dimension of the input data feature')
    parser.add_argument('--latent_size', type=int, default=100, help='Dimension of the latent space')
    parser.add_argument('--batch_size', type=int, default=512)
    
    parser.add_argument('--lr', type=float, default=1.0e-04)
    parser.add_argument('--n_epochs', type=int, default=100)

    # parser.add_argument('--train_site', nargs='+', default=['a', 'b'])
    parser.add_argument('--train_site', nargs='+', default=['a'])
    parser.add_argument('--val_site', type=str, choices=['a', 'b', 'c'], default='a')
    parser.add_argument('--log_dir', type=str, default='./log')
    parser.add_argument('--comments', type=str, default='USAD_Lightning')

    return parser.parse_args()


def main():
    # pl.seed_everything(42)
    args = parse_args()
    # args.log_dir = os.path.join(args.log_dir, args.comments, datetime.datetime.now().strftime("%m%d%H%M%S"))
    # print('=' * 50)
    # print('[info msg] arguments')
    # for key, value in vars(args).items():
    #     print(key, ":", value)
    # print('=' * 50)

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
    model = to_device(model, 'mps')
    history = training(args.n_epochs, model, train_loader, train_loader)

if __name__ == '__main__':
    main()
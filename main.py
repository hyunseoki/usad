from usad import Encoder, Decoder
import argparse, os, datetime
import lightning.pytorch as pl
import torch


def main():
    # batch_size = 128
    # signal_length = 320
    # in_shape = (batch_size, signal_length, 1)
    # hidden_size = 100

    # w_size = in_shape[1] * in_shape[2]
    # z_size = in_shape[1] * hidden_size
    e = Encoder(in_size=320, latent_size=100)
    x = torch.randn(4, 320)
    print(e(x).shape)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    
    # --- USAD 관련 하이퍼파라미터 추가 ---
    parser.add_argument('--input_size', type=int, default=320, help='Dimension of the input data feature')
    parser.add_argument('--latent_size', type=int, default=100, help='Dimension of the latent space')
    parser.add_argument('--batch_size', type=int, default=128)
    
    parser.add_argument('--lr', type=float, default=1.0e-04)
    parser.add_argument('--n_epochs', type=int, default=100)

    parser.add_argument('--train_site', nargs='+', default=['a', 'b'])
    parser.add_argument('--val_site', type=str, choices=['a', 'b', 'c'], default='c')
    parser.add_argument('--log_dir', type=str, default='./log')
    parser.add_argument('--comments', type=str, default='USAD_Lightning')

    return parser.parse_args()


if __name__ == '__main__':
    # pl.seed_everything(42)
    # args = parse_args()
    # args.log_dir = os.path.join(args.log_dir, args.comments, datetime.datetime.now().strftime("%m%d%H%M%S"))
    # print('=' * 50)
    # print('[info msg] arguments')
    # for key, value in vars(args).items():
    #     print(key, ":", value)
    # print('=' * 50)

    main()
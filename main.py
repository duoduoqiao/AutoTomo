import os
import argparse
import numpy as np

from pathlib import Path
from net_params import AutoTomo
from data_utils import load_data
from plot_utils import plot_sp, plot_tm
from train_utils import Trainer_autotomo


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment", add_help=False)

    parser.add_argument('--root_dir', type=str, default='./',
                        help='Root directory')
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size")
    parser.add_argument('--patience', type=int, default=50,
                        help="Patience")
    parser.add_argument('--epoch', type=int, default=500,
                        help="Epoch")
    parser.add_argument('--loss_func', type=str, default="l1norm", 
                        choices=["l1norm", "mse"],
                        help="Loss function")
    parser.add_argument('--mode', type=str, default="all",
                        choices=["all", "unobserved"],
                        help="Show all flows in test_set or just unobserved flows in train_set")
    parser.add_argument('--dataset', type=str, default="abilene",
                        choices=["abilene", "geant", "cernet"],
                        help="Dataset")
    parser.add_argument('--model', type=str, default='autotomo',
                        choices=['autotomo', 'autotomo_os'], 
                        help="Choose Model")
    parser.add_argument('--unknown_rate', type=float, default=0.1,
                        choices=[i*0.1 for i in range(8)],
                        help="Unknown rate")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")

    parser.add_argument('--hd_1', type=int, default=80, help="Hidden dimensions 1st")
    parser.add_argument('--hd_2', type=int, default=100, help="Hidden dimensions 2nd")

    args = parser.parse_args()
    return args


def main(args):
    if args.dataset == 'abilene':
        tm_fn = './Data/Abilene/abilene_tm.csv'
        rm_fn = './Data/Abilene/abilene_rm.csv'
        scale = 10**9
        n_train =  15*7
        n_test = 1*7
    elif args.dataset == 'cernet':
        tm_fn = './Data/Abilene/cernet_tm.csv'
        rm_fn = './Data/Abilene/cernet_rm.csv'
        scale = 10**7
        n_train =  4*7
        n_test = 1*7
    else:
        tm_fn = './Data/GEANT/geant_tm.csv'
        rm_fn = './Data/GEANT/geant_rm.csv'
        scale = 10**7
        n_train =  10*7
        n_test = 1*7

    train_loader, test_loader, rm, known_id, real_train_loader =\
        load_data(root_dir=args.root_dir, tm_fn=tm_fn, rm_fn=rm_fn, n_train=n_train, n_test=n_test,
                  known_rate=1.-args.unknown_rate, b_size=args.batch_size, scale=scale, 
                  model_name=args.model, dataset_name=args.dataset)
    
    INPUT_SIZE, OUTPUT_SIZE = train_loader.dataset.link_dim, train_loader.dataset.feat_dim

    model = AutoTomo(INPUT_SIZE, args.hd_1, args.hd_2, OUTPUT_SIZE)
    print(model)

    trainer = Trainer_autotomo(model, rm, train_loader, train_lr=args.lr, known_id=known_id, train_epoch=args.epoch,
                               batch_size=args.batch_size, patience=args.patience, loss_type=args.loss_func, 
                               weight=1.-args.unknown_rate)

    trainer.train()
    # trainer.load()

    if args.mode == "unobserved":
            unknown_train_id = np.setdiff1d(np.arange(OUTPUT_SIZE), known_id.cpu().numpy(), assume_unique=True)
            pred_train_flow = trainer.estimate(data_loader=real_train_loader)
            pred_flow, real_flow = pred_train_flow[:, unknown_train_id],\
                                   real_train_loader.dataset.label.cpu().numpy()[:, unknown_train_id]
    elif args.mode == "all":
            pred_flow = trainer.estimate(data_loader=test_loader)
            real_flow = test_loader.dataset.label.cpu().numpy()

    if not Path("Figures").exists():
        os.mkdir("Figures")
    plot_sp(real_flow, pred_flow, args.model, args.loss_func, os.path.join("Figures", args.mode+"_error_sp.jpg"))
    plot_tm(real_flow, pred_flow, args.model, args.loss_func, os.path.join("Figures", args.mode+"_error_tm.jpg"))


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)

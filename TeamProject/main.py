import warnings
warnings.filterwarnings("ignore")

from dataloader import *
from models import *
from train import *

import argparse
import os

import torch

def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('device:', device)

    save_path = f'./outputs/model_del_feature={args.del_feature}'
    os.makedirs(save_path, exist_ok=True)

    train_dataset = make_dataset('train', args.T, args.t, args.del_feature, device)
    val_dataset = make_dataset('val', args.T, args.t, args.del_feature, device)

    fix_seed(42)
    model = TemporalGNN(
    lst_channels = [train_dataset[0][0].x.shape[1], 64, 128], # first dimension = # of features of the low-level graph
    second_dim = train_dataset[0][1].x.shape[1], # second dimension = # of features of the high-level graph
    T = args.T,
    t = args.t,
    address_start = torch.load('./data/address_start.pt').to(torch.long),
    address_end = torch.load('./data/address_end.pt').to(torch.long),
    device = device
    ).to(device)
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1, eta_min=0.00001)

    fix_seed(42)
    train_GNN(model, train_dataset, val_dataset, optimizer, criterion, scheduler, save_path, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", default = 42, type = int)
    parser.add_argument("-t", default = 1, type = int)
    parser.add_argument("-T", default = 24, type = int)
    parser.add_argument("-d", "--del_feature", choices = ['None' , 'day_of_week', 'lane_count', 'road_rating', 'road_type', 
                                                          'end_turn_restricted', 'start_latitude', 'end_latitude', 'rain', 'temp',
                                                          'humid', 'wind'], default = 'None')
    parser.add_argument("--lr", default = 0.001, type = float)
    parser.add_argument("-e", "--epoch", default = 100, type = int)
    args = parser.parse_args()

    main(args)
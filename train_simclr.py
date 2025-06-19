from argparse import ArgumentParser
import glob
import os
import random

import torch
import torch.nn as nn

from lightly import loss as losses
from lightly import transforms
from lightly.data import LightlyDataset
from lightly.models.modules import heads

from tqdm import tqdm

import yaml

from core.utils.factories import get_encoder, get_decoder, get_optimizer
from core.utils.utils import get_parameter_count, get_train_test_split, get_run_path, save_latest, save_best


class SimCLR(nn.Module):
    def __init__(self, backbone, input_dim, hidden_dim, output_dim):
        super(SimCLR, self).__init__()
        self.backbone = backbone
        self.projection_head = heads.SimCLRProjectionHead(
            input_dim = input_dim,
            hidden_dim = hidden_dim,
            output_dim = output_dim
        )
    
    def forward(self, x):
        features = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(features)
        return z

def main(args, progress):
    device = torch.device('cuda:0') if torch.cuda.device_count() > 0 else torch.device('cpu')
    vision_cfg = args['vision_encoder']
    sim_clr_cfg = args['simclr']
    optimizer_cfg = args['optimizer']
    hyperparameter_cfg = args['hyperparameters']

    if 'seed' in hyperparameter_cfg:
        random.seed(hyperparameter_cfg['seed'])
    
    transform = transforms.SimCLRTransform(input_size=sim_clr_cfg['image_size'])

    filenames = [os.path.basename(x) for x in glob.glob(os.path.join('.','output','base','*.png'))]
    random.shuffle(filenames)
    train_data, test_data = get_train_test_split(filenames, hyperparameter_cfg['train_split_size'])
    train_dataset = LightlyDataset(input_dir=os.path.join('.','output','base'), filenames=train_data, transform=transform)
    test_dataset = LightlyDataset(input_dir=os.path.join('.','output','base'), filenames=test_data, transform=transform)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = 16,
        shuffle = True
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size = 16,
        shuffle = False
    )

    vision_encoder = get_encoder(vision_cfg)
    simclr_trainer = SimCLR(
        backbone = vision_encoder,
        input_dim = sim_clr_cfg['input_dim'],
        hidden_dim = sim_clr_cfg['hidden_dim'],
        output_dim = sim_clr_cfg['output_dim']
    ).to('cuda')
    print('Number of parameters:\n\tTrainable: {}\n\tUntrainable: {}'.format(*(get_parameter_count(simclr_trainer))))
    optimizer = get_optimizer(optimizer_cfg, simclr_trainer.parameters())

    transform = transforms.SimCLRTransform(
        input_size = sim_clr_cfg['image_size']
    )

    criterion = losses.NTXentLoss(temperature=0.5)

    run_path = get_run_path(run_name='simclr')
    print('Run path: {}'.format(run_path))

    with open(os.path.join(run_path, 'config.yaml'), 'w') as f:
        yaml.dump(args, f)
    
    epochs = hyperparameter_cfg['epochs']
    max_patience = hyperparameter_cfg['max_patience']
    patience = 0
    best_loss = float('inf')

    train_csv, test_csv = open(os.path.join(run_path, 'train.csv'), '+w'), open(os.path.join(run_path, 'test.csv'), '+w')
    train_csv.write('epoch,train_loss\n')
    test_csv.write('epoch,test_loss\n')

    for epoch in range(10):
        simclr_trainer.train()
        total_loss = 0
        for (x0, x1), _, _ in tqdm(train_dataloader, leave=False):
            z0 = simclr_trainer(x0.to('cuda'))
            z1 = simclr_trainer(x1.to('cuda'))
            loss = criterion(z0, z1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        train_csv.write(f'{epoch},{total_loss/len(train_dataloader)}\n')
        print(f"Epoch {epoch}: Train Loss: {total_loss/len(train_dataloader)}")

        simclr_trainer.eval()
        total_loss = 0
        with torch.no_grad():
            for (x0, x1), _, _ in tqdm(test_dataloader, leave=False):
                z0 = simclr_trainer(x0.to('cuda'))
                z1 = simclr_trainer(x1.to('cuda'))
                loss = criterion(z0, z1)
                total_loss += loss.item()
        test_csv.write(f'{epoch},{total_loss/len(test_dataloader)}\n')
        if total_loss < best_loss:
            patience = 0
        else:
            patience += 1
            if patience >= max_patience:
                print(f"Early stopping at epoch {epoch}")
                break
        print(f"Epoch {epoch}: Test Loss: {total_loss/len(test_dataloader)}; Patience: {patience}/{max_patience}")
        save_latest(run_path, simclr_trainer, optimizer, epoch)
        if total_loss < best_loss:
            torch.save(simclr_trainer.backbone.state_dict(), os.path.join(run_path, 'image_encoder.pth'))
        best_loss = save_best(run_path, simclr_trainer, optimizer, epoch, total_loss, best_loss)


if __name__ == '__main__':
    parser = ArgumentParser(description='Train SimCLR model')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    parser.add_argument('--progress', action='store_true', help='Show progress bars')
    args = parser.parse_args()
    # Load configuration file
    config_path = args.config
    if not os.path.isfile(config_path):
        raise ValueError(f"Configuration file '{config_path}' does not exist.")
    with open(config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ValueError(f"Error parsing YAML configuration file: {exc}")
    main(config, args.progress)
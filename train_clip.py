from argparse import ArgumentParser
import os
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm

import yaml

from core.datasets.sa_dataset import get_data, SADataset
from core.models.other.clip import CLIPTraininer
from core.utils.factories import get_encoder, get_decoder, get_optimizer
from core.utils.utils import get_parameter_count, get_train_test_split, get_run_path, save_latest, save_best
from core.utils.zclip import ZClip

def main(args):
    device = torch.device('cuda:0') if torch.cuda.device_count() > 0 else torch.device('cpu')
    vision_cfg = args['vision_encoder']
    text_cfg = args['text_encoder']
    optimizer_cfg = args['optimizer']
    hyperparameter_cfg = args['hyperparameters']

    if 'seed' in hyperparameter_cfg:
        random.seed(hyperparameter_cfg['seed'])

    print('Loading base data')
    vocab, data = get_data(max_len=args['text_encoder']['args']['max_saml_layers'])
    print('Loading basic synthetic data')
    _, syn_data = get_data(max_len=args['text_encoder']['args']['max_saml_layers'], data_path=os.path.join('.','output','synthetic','basic'))
    text_cfg['args']['vocab_size'] = len(vocab)

    random.shuffle(data)
    train_data, test_data = get_train_test_split(data, hyperparameter_cfg['train_split_size'])
    train_data += syn_data
    train_dataset, test_dataset = SADataset(train_data, transforms=True), SADataset(test_data, transforms=False)
    train_dataloader = DataLoader(train_dataset, batch_size=hyperparameter_cfg['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=hyperparameter_cfg['batch_size'], shuffle=False)
    print('Train dataset size: {}'.format(len(train_dataset)))
    print('Test dataset size: {}'.format(len(test_dataset)))

    vision_encoder = get_encoder(vision_cfg)
    text_encoder = get_encoder(text_cfg)
    clip_trainer = CLIPTraininer(vision_encoder, text_encoder).to(device)
    print('Number of parameters:\n\tTrainable: {}\n\tUntrainable: {}'.format(*(get_parameter_count(clip_trainer))))
    optimizer = get_optimizer(optimizer_cfg, clip_trainer.parameters())
    zclip = ZClip()

    run_path = get_run_path(run_name='clip')
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

    for epoch in range(epochs):
        clip_trainer.train()
        total_loss = 0.0
        for bdx, batch in enumerate(tqdm(train_dataloader, desc='Training', leave=False)):
            feature = batch['feature'].to(device)
            labels = batch['label'].to(device)
            masks = batch['mask'].to(device)

            batch_logits = clip_trainer(feature, labels, masks)
            labels_idx = 2 * torch.eye(batch_logits.shape[0]).to(device) - 1
            loss = -torch.sum(F.logsigmoid(labels_idx * batch_logits)) / batch_logits.shape[0]

            optimizer.zero_grad()
            loss.backward()
            zclip.step(clip_trainer)
            optimizer.step()

            total_loss += loss.item()
        train_csv.write(f'{epoch},{total_loss/len(train_dataloader)}\n')
        print(f"Epoch {epoch}: Train Loss: {total_loss/len(train_dataloader)}")

        clip_trainer.eval()
        total_loss = 0.0
        with torch.no_grad():
            for bdx, batch in enumerate(tqdm(test_dataloader, desc='Testing', leave=False)):
                feature = batch['feature'].to(device)
                labels = batch['label'].to(device)
                masks = batch['mask'].to(device)

                batch_logits = clip_trainer(feature, labels, masks)
                labels_idx = 2 * torch.eye(batch_logits.shape[0]).to(device) - 1
                loss = -torch.sum(F.logsigmoid(labels_idx * batch_logits)) / batch_logits.shape[0]

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
        save_latest(run_path, clip_trainer, optimizer, epoch)
        if total_loss < best_loss:
            torch.save(clip_trainer.saml_encoder.state_dict(), os.path.join(run_path, 'saml_encoder.pth'))
            torch.save(clip_trainer.image_encoder.state_dict(), os.path.join(run_path, 'image_encoder.pth'))
        best_loss = save_best(run_path, clip_trainer, optimizer, epoch, total_loss, best_loss)
    train_csv.close()
    test_csv.close()


if __name__ == '__main__':
    parser = ArgumentParser(description='Train CLIP model')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
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
    main(config)
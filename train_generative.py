from argparse import ArgumentParser
import glob
import os
import random

from einops import repeat

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import transforms

from tqdm import tqdm

import yaml

from core.datasets.sa_dataset import get_data, SADataset
from core.models.other.saml_generator import SAMLGenerator
from core.utils.factories import get_encoder, get_decoder, get_optimizer
from core.utils.utils import get_parameter_count, get_train_test_split, get_run_path, save_latest, save_best, read_img_cv2, convert_numpy_to_saml
from core.utils.zclip import ZClip

def load_test_imgs(test_loc):
    test_imgs = []
    for img_loc in glob.glob(os.path.join(test_loc, '*.png')):
        img_name = os.path.basename(img_loc)[:-4]
        feature = torch.tensor(read_img_cv2(img_loc) / 255.).permute(2, 0, 1)
        feature = transforms.Resize((256, 256), antialias=True)(feature)
        test_imgs.append({'feature': feature, 'name': img_name})
    return test_imgs

def convert_saml_numpy(saml_layers):
    for ldx in range(len(saml_layers)):
        saml_layers[ldx][1] *= 255
        saml_layers[ldx][2] *= 255
        saml_layers[ldx][3] *= 255
        saml_layers[ldx][5] *= 127
        saml_layers[ldx][6] *= 127
        saml_layers[ldx][7] *= 127
        saml_layers[ldx][8] *= 127
        saml_layers[ldx][9] *= 127
        saml_layers[ldx][10] *= 127
        saml_layers[ldx][11] *= 127
        saml_layers[ldx][12] *= 127
    return saml_layers

def main(args, progress):
    device = torch.device('cuda:0') if torch.cuda.device_count() > 0 else torch.device('cpu')
    vision_cfg = args['vision_encoder']
    vision_weights = None
    if 'checkpoint' in vision_cfg:
        vision_weights = torch.load(os.path.join(vision_cfg['checkpoint'], 'image_encoder.pth'))
        with open(os.path.join(vision_cfg['checkpoint'], 'config.yaml'), 'r') as f:
            vision_cfg = yaml.safe_load(f)['vision_encoder']
    text_cfg = args['text_decoder']
    text_cfg['args']['input_dim'] = vision_cfg['args']['dim']
    optimizer_cfg = args['optimizer']
    hyperparameter_cfg = args['hyperparameters']

    if 'seed' in hyperparameter_cfg:
        random.seed(hyperparameter_cfg['seed'])
    
    print('Loading base data')
    vocab, data = get_data(max_len=args['text_decoder']['args']['max_saml_layers'], verbose=progress)
    print('Loading low layer synthetic data')
    vocab, sin_data = get_data(max_len=args['text_decoder']['args']['max_saml_layers'], vocab=vocab, verbose=progress, data_path=os.path.join('.','output','synthetic','single'))
    print('Loading basic synthetic data')
    vocab, bas_data = get_data(max_len=args['text_decoder']['args']['max_saml_layers'], vocab=vocab, verbose=progress, data_path=os.path.join('.','output','synthetic','basic'))
    text_cfg['args']['vocab_size'] = len(vocab)

    random.shuffle(data)
    train_data, test_data = get_train_test_split(data, hyperparameter_cfg['train_split_size'])
    train_data += sin_data
    train_data += bas_data
    train_dataset, test_dataset = SADataset(train_data, transforms=True), SADataset(test_data, transforms=False)
    train_dataloader = DataLoader(train_dataset, batch_size=hyperparameter_cfg['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=hyperparameter_cfg['batch_size'], shuffle=False)
    print('Train dataset size: {}'.format(len(train_dataset)))
    print('Test dataset size: {}'.format(len(test_dataset)))

    vision_encoder = get_encoder(vision_cfg)
    if vision_weights is not None:
        vision_encoder.load_state_dict(vision_weights)
    text_encoder = get_decoder(text_cfg)
    model = SAMLGenerator(vision_encoder, text_encoder).to(device)
    print('Number of parameters:\n\tTrainable: {}\n\tUntrainable: {}'.format(*(get_parameter_count(model))))
    optimizer = get_optimizer(optimizer_cfg, model.parameters())
    zclip = ZClip()

    run_path = get_run_path(run_name='saml')
    print('Run path: {}'.format(run_path))

    with open(os.path.join(run_path, 'config.yaml'), 'w') as f:
        yaml.dump(args, f)

    test_imgs = load_test_imgs(hyperparameter_cfg['test_data'])
    
    epochs = hyperparameter_cfg['epochs']
    max_patience = hyperparameter_cfg['max_patience']
    if 'unfreeze_epoch' in hyperparameter_cfg:
        unfreeze_epoch = int(hyperparameter_cfg['unfreeze_epoch'])
    else:
        unfreeze_epoch = 0
    patience = 0
    best_loss = float('inf')

    train_csv, test_csv = open(os.path.join(run_path, 'train.csv'), '+w'), open(os.path.join(run_path, 'test.csv'), '+w')
    train_csv.write('epoch,train_loss\n')
    test_csv.write('epoch,test_loss\n')

    for epoch in range(epochs):
        model.train()
        if epoch == unfreeze_epoch:
            model.unfreeze_encoder()
            print('Unfreezing encoder')
        elif epoch < unfreeze_epoch:
            model.freeze_encoder()
        total_loss, cls_total, col_total, pos_total = 0.0, 0.0, 0.0, 0.0
        for bdx, batch in enumerate(tqdm(train_dataloader, desc='Training', leave=False, disable=not progress)):
            feature = batch['feature'].to(device)
            labels = batch['label']
            xin, xout = labels[:, :-1].to(device), labels[:, 1:].to(device)
            cls_out, col_out, pos_out = xout[:,:,0], xout[:,:,1:5], xout[:,:,5:]
            mask_in, mask_out = batch['mask'][:, :-1].to(device), batch['mask'][:, 1:].to(device)
            cls_guess, col_guess, pos_guess = model(xin, feature, mask_in)
            cls_loss = F.cross_entropy(cls_guess.permute(0,2,1), cls_out.long(), ignore_index=vocab['<PAD>'])
            cls_total += cls_loss.item()
            col_loss = F.smooth_l1_loss(col_guess, col_out, reduction='none')
            col_loss = (col_loss * repeat(mask_out, 'b l -> b l d', d=4).float()).sum()
            col_loss = col_loss / mask_out.sum()
            col_total += col_loss.item()
            pos_loss = F.smooth_l1_loss(pos_guess, pos_out, reduction='none')
            pos_loss = (pos_loss * repeat(mask_out, 'b l -> b l d', d=8).float()).sum()
            pos_loss = pos_loss / mask_out.sum()
            pos_total += pos_loss.item()
            train_loss = cls_loss + col_loss + pos_loss
            optimizer.zero_grad()
            train_loss.backward()
            zclip.step(model)
            optimizer.step()

            total_loss += train_loss.item()
        train_csv.write(f'{epoch},{total_loss/len(train_dataloader)}\n')
        print(f"Epoch #{epoch}; Train Loss: {total_loss/len(train_dataloader)}; Class Loss: {cls_total/len(train_dataloader)}; Color Loss: {col_total/len(train_dataloader)}; Position Loss: {pos_total/len(train_dataloader)}")

        model.eval()
        total_loss, cls_total, col_total, pos_total = 0.0, 0.0, 0.0, 0.0
        with torch.no_grad():
            for bdx, batch in enumerate(tqdm(test_dataloader, desc='Testing', leave=False, disable=not progress)):
                feature = batch['feature'].to(device)
                labels = batch['label']
                xin, xout = labels[:, :-1].to(device), labels[:, 1:].to(device)
                cls_out, col_out, pos_out = xout[:,:,0], xout[:,:,1:5], xout[:,:,5:]
                mask_in, mask_out = batch['mask'][:, :-1].to(device), batch['mask'][:, 1:].to(device)

                cls_guess, col_guess, pos_guess = model(xin, feature, mask_in)
                cls_loss = F.cross_entropy(cls_guess.permute(0,2,1), cls_out.long(), ignore_index=vocab['<PAD>'])

                cls_total += cls_loss.item()
                col_loss = F.smooth_l1_loss(col_guess, col_out, reduction='none')
                col_loss = (col_loss * repeat(mask_out, 'b l -> b l d', d=4).float()).sum()
                col_loss = col_loss / mask_out.sum()
                col_total += col_loss.item()
                pos_loss = F.smooth_l1_loss(pos_guess, pos_out, reduction='none')
                pos_loss = (pos_loss * repeat(mask_out, 'b l -> b l d', d=8).float()).sum()
                pos_loss = pos_loss / mask_out.sum()
                pos_total += pos_loss.item()
                test_loss = (cls_loss + col_loss + pos_loss)
                total_loss += test_loss.item()
        test_csv.write(f'{epoch},{total_loss/len(test_dataloader)}\n')
        if total_loss < best_loss:
            patience = 0
        else:
            patience += 1
            if patience >= max_patience:
                print(f"Early stopping at epoch {epoch}")
                break
        print(f"Epoch {epoch}: Test Loss: {total_loss/len(test_dataloader)}; Class Loss: {cls_total/len(test_dataloader)}; Color Loss: {col_total/len(test_dataloader)}; Position Loss: {pos_total/len(test_dataloader)}; Patience: {patience}/{max_patience}")
        save_latest(run_path, model, optimizer, epoch)
        if total_loss < best_loss:
            for img in test_imgs:
                img_feat = img['feature'].to(device)
                img_name = img['name']
                saml_layers = model.generate(img_feat.unsqueeze(0), vocab, device)
                saml_layers = convert_saml_numpy(saml_layers)
                convert_numpy_to_saml(saml_layers, os.path.join(run_path, img_name + '.saml'), name=img_name)
            print(f"Epoch {epoch}: Generated SAML layers saved.")
        best_loss = save_best(run_path, model, optimizer, epoch, total_loss, best_loss)
    train_csv.close()
    test_csv.close()


if __name__ == '__main__':
    parser = ArgumentParser(description='Train CLIP model')
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
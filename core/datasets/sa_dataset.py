import glob
import os
import random
import xml.etree.ElementTree as ETO

import lxml.etree as ET

import numpy as np

import torch
from torch.utils.data import Dataset

from torchvision import transforms

from tqdm import tqdm

from webcolors import hex_to_rgb, rgb_to_hex

from core.datasets.vocab import Vocabulary
from core.utils.utils import read_img_cv2, clamp

class SADataset(Dataset):
    def __init__(self, data, cache_data = True, transforms = False):
        self.data = data
        self.max_len = 256
        self.cache_data = cache_data
        self.transforms = transforms
        if cache_data:
            self.features = [None] * len(data)
            self.labels = [None] * len(data)
            self.masks = [None] * len(data)
        else:
            self.features = None
            self.labels = None
            self.masks = None
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        if self.cache_data and self.features[idx] != None:
            feature = self.features[idx]
            label = self.labels[idx]
            mask = self.masks[idx]
            return {'feature': feature, 'label': label, 'mask': mask}

        cur_data = self.data[idx]

        feature = torch.tensor(read_img_cv2(cur_data['feature']) / 255.).permute(2, 0, 1)
        feature = transforms.Resize((256, 256), antialias=True)(feature)
        label = torch.tensor(cur_data['label'])
        mask = torch.tensor(cur_data['mask'], dtype=torch.bool)

        if self.cache_data:
            self.features[idx] = feature
            self.labels[idx] = label
            self.masks[idx] = mask
        
        if self.transforms:
            feature, label = self.random_mod(feature, label)

        return {'feature': feature, 'label': label, 'mask': mask}
    
    def random_mod(self, feature, label):
        if random.random() > 0.5:
            feature = transforms.RandomVerticalFlip(1.0)(feature)
            label[:, 6] *= -1
            label[:, 8] *= -1
            label[:, 10] *= -1
            label[:, 12] *= -1
        if random.random() > 0.5:
            feature = transforms.RandomHorizontalFlip(1.0)(feature)
            label[:, 5] *= -1
            label[:, 7] *= -1
            label[:, 9] *= -1
            label[:, 11] *= -1
        return feature, label

def get_data(verbose: bool = False, vocab: Vocabulary = None, layer_names: list = None, data_path: str = os.path.join('.','output','base'), max_len: int = 256, pos_const: float = 127.0, col_const: float = 256.0):
    all_data = glob.glob(os.path.join(data_path, '*.saml'))
    data = []
    vocab = Vocabulary(layer_names=layer_names) if vocab is None else vocab
    sos_line = [vocab['<SOS>']] + [0] * 12
    eos_line = [vocab['<EOS>']] + [0] * 12
    pad_line = [vocab['<PAD>']] + [0] * 12
    
    for saml_path in tqdm(all_data, desc='Loading SAML files', leave=False, disable=not verbose):
        try:
            with open(saml_path, 'r', encoding='utf-8-sig') as f:
                root = ETO.fromstring(f.read())
            
            layers = [sos_line]
            mask = [True]
            
            # New function to process elements recursively
            def process_elements(element):
                collected_layers = []
                for child in element:
                    if child.tag == 'layer':
                        # Process layer as before
                        attribs = child.attrib
                        cur_type = vocab[attribs['type']]
                        if cur_type == '0':
                            print(cur_type)
                        r, g, b = hex_to_rgb(attribs['color'])
                        ltx, lty, lbx, lby, rtx, rty, rbx, rby = list(map(
                            lambda x: int(x)/pos_const, 
                            [attribs['ltx'], attribs['lty'], attribs['lbx'], 
                             attribs['lby'], attribs['rtx'], attribs['rty'], 
                             attribs['rbx'], attribs['rby']]
                        ))
                        r, g, b, a = r/col_const, g/col_const, b/col_const, float(attribs['alpha'])
                        collected_layers.append([cur_type, r, g, b, a, ltx, lty, lbx, lby, rtx, rty, rbx, rby])
                    elif child.tag == 'g':
                        # Recursively process group
                        collected_layers.extend(process_elements(child))
                return collected_layers
            
            # Get all layers from all groups and direct children
            all_layers = process_elements(root)
            
            # Add all collected layers
            for layer_data in all_layers:
                layers.append(layer_data)
                mask.append(True)
                
            layers.append(eos_line)
            mask.append(True)
            while len(layers) < max_len:
                layers.append(pad_line)
                mask.append(False)
                
            img_path = saml_path[:-4] + 'png'
            bundle = {'feature': img_path, 'label': np.asarray(layers, dtype=np.float32), 'mask': np.asarray(mask, dtype=bool)}
            data.append(bundle)
        except Exception as e:
            print(f'Error loading {saml_path}: {e}')
    return vocab, data

def convert_numpy_to_saml(data: list, vocab: Vocabulary, dest_path: str = None, name: str = 'Test', clamp_values: bool = False, pos_const: float = 127.0, col_const: float = 256.0) -> None:
    if dest_path is None:
        dest_path = name + '.saml'
    
    with open(dest_path, 'w') as f:
        f.write('<?xml version="1.0" encoding="utf-8"?>\n')
        saml_lines = []
        for line in data:
            if line[0] == vocab['<SOS>'] or line[0] == vocab['<PAD>'] or line[0] == vocab['<EOS>']:
                continue
            else:
                saml_lines.append(line)
        xml_data = ET.Element('sa')
        xml_data.set('name', name)
        xml_data.set('visible', 'true')
        xml_data.set('version', '1')
        xml_data.set('author', '1337')
        xml_data.set('width', '192')
        xml_data.set('height', '96')
        xml_data.set('sound', '1')
        for ldx, line in enumerate(saml_lines):
            layer = ET.SubElement(xml_data, 'layer')
            layer.set('name', 'Symbol {}'.format(ldx))
            layer.set('visible', 'true')
            layer.set('type', '{}'.format(vocab[int(line[0])]))
            color_tup = [clamp(int(x * col_const), 0, 255) for x in line[1:4]]
            color_tup = rgb_to_hex(color_tup)
            layer.set('color', str(color_tup))
            alpha_val = '{:.6f}'.format(clamp(line[4], 0, 1))
            layer.set('alpha', alpha_val)
            positions = list(map(lambda x: str(clamp(int(((x * pos_const))), -127, 127)), line[5:]))
            layer.set('ltx', positions[0])
            layer.set('lty', positions[1])
            layer.set('lbx', positions[2])
            layer.set('lby', positions[3])
            layer.set('rtx', positions[4])
            layer.set('rty', positions[5])
            layer.set('rbx', positions[6])
            layer.set('rby', positions[7])
        f.write(ET.tostring(xml_data, pretty_print=True).decode('utf8'))
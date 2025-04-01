import glob
import os
import shutil
import xml.etree.ElementTree as ETO

import cv2

import numpy as np

from tqdm import tqdm

from webcolors import hex_to_rgb

from core.datasets.layerwise import SAImage

def get_data_for_conversion(path: str):
    all_data = glob.glob(os.path.join(path, '*.saml'))
    data = []
    for saml_path in tqdm(all_data, desc='Loading SAML files', leave=False):
        with open(saml_path, 'r', encoding='utf-8-sig') as f:
            root = ETO.fromstring(f.read())
        layers = []
        names = []
        for edx, layer in enumerate(root):
            attribs = layer.attrib
            cur_type = attribs['type']
            r, g, b = hex_to_rgb(attribs['color'])
            ltx, lty, lbx, lby, rtx, rty, rbx, rby = list(map(lambda x: int(x), [attribs['ltx'], attribs['lty'], attribs['lbx'], attribs['lby'], attribs['rtx'], attribs['rty'], attribs['rbx'], attribs['rby']]))
            r, g, b, a = r, g, b, float(attribs['alpha'])
            layers.append([cur_type, r, g, b, a, ltx, lty, lbx, lby, rtx, rty, rbx, rby])
        name = os.path.splitext(os.path.basename(saml_path))[0]
        layers = list(reversed(layers))
        bundle = {'layers': layers, 'name': name}
        data.append(bundle)
    return data

def main(data_folder: str, output_folder = None):
    raw_data = get_data_for_conversion(data_folder)
    if output_folder is None:
        output_folder = data_folder
    for bundle in tqdm(raw_data, desc='Converting SAML files', leave=False):
        layers = bundle['layers']
        name = bundle['name']
        sa_image = SAImage(layers)
        img_array = sa_image.render()
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGRA)
        cv2.imwrite(os.path.join(output_folder, name + '.png'), img_array)
        shutil.copyfile(os.path.join(data_folder, name + '.saml'), os.path.join(output_folder, name + '.saml'))


if __name__ == '__main__':
    main('data', output_folder=os.path.join('output', 'base'))
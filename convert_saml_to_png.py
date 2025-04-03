import glob
import os
import shutil
import xml.etree.ElementTree as ETO

import cv2

import numpy as np

from tqdm import tqdm

from webcolors import hex_to_rgb

from core.datasets.layerwise import SAImage
from core.utils.utils import get_data_for_conversion

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
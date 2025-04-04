import glob
import os
import random
import shutil
from typing import List, Any
import xml.etree.ElementTree as ETO

import cv2

import numpy as np

from tqdm import tqdm

from webcolors import hex_to_rgb

from core.datasets.layerwise import Layer, SAImage
from core.utils.utils import get_data_for_conversion, convert_numpy_to_saml, clamp

def random_name(length: int = 10) -> str:
    return ''.join(random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(length))

def remove_introns(img_layers: list):
    
    return img_layers

def crossover(img_layers_one: list, img_layers_two: list) -> list:
    crossover_point = random.randint(0, min(len(img_layers_one) - 1, len(img_layers_two) - 1))
    img_one_part_one, _ = img_layers_one[:crossover_point], img_layers_one[crossover_point:]
    _, img_two_part_two = img_layers_two[:crossover_point], img_layers_two[crossover_point:]
    return img_one_part_one + img_two_part_two

def mutate(img_layers: list) -> list:
    for ldx in range(len(img_layers)):
        img_layers[ldx][2] = clamp(img_layers[ldx][2] + random.randint(-5, 5), 0, 255)
        img_layers[ldx][3] = clamp(img_layers[ldx][3] + random.randint(-5, 5), 0, 255)
        img_layers[ldx][4] = clamp(img_layers[ldx][4] + random.randint(-5, 5), 0, 255)
        img_layers[ldx][5] = clamp(img_layers[ldx][5] + random.uniform(-0.1, 0.1), 0, 1)
        img_layers[ldx][6] = clamp(img_layers[ldx][6] + random.randint(-5, 5), -127, 127)
        img_layers[ldx][7] = clamp(img_layers[ldx][7] + random.randint(-5, 5), -127, 127)
        img_layers[ldx][8] = clamp(img_layers[ldx][8] + random.randint(-5, 5), -127, 127)
        img_layers[ldx][9] = clamp(img_layers[ldx][9] + random.randint(-5, 5), -127, 127)
        img_layers[ldx][10] = clamp(img_layers[ldx][10] + random.randint(-5, 5), -127, 127)
        img_layers[ldx][11] = clamp(img_layers[ldx][11] + random.randint(-5, 5), -127, 127)
        img_layers[ldx][12] = clamp(img_layers[ldx][12] + random.randint(-5, 5), -127, 127)
        img_layers[ldx][13] = clamp(img_layers[ldx][13] + random.randint(-5, 5), -127, 127)
    return img_layers

def evolution_round(img_layers_one: list, img_layers_two: list, crossover_prob: float = 0.3, mutation_prob: float = 0.7):
    if random.random() < crossover_prob:
        img_layers_one = crossover(img_layers_one, img_layers_two)
    if random.random() < mutation_prob:
        img_layers_one = mutate(img_layers_one)
    return img_layers_one

def main(data_folder: str, output_folder = None, num_rounds: int = 10, num_to_generate: int = 100):

    data = get_data_for_conversion(data_folder)
    for _ in tqdm(range(num_to_generate), desc='Generating images', leave=False):
        layers_one, layers_two = random.choices(data, k=2)
        layers_one = layers_one['layers']
        layers_two = layers_two['layers']
        for _ in range(num_rounds):
            layers_one = evolution_round(layers_one, layers_two)
        cleaned_one = remove_introns(layers_one)
        name = random_name()
        convert_numpy_to_saml(list(reversed(cleaned_one)), os.path.join(output_folder, name + '.saml'), name=name)

if __name__ == '__main__':
    main(os.path.join('output', 'base'), output_folder=os.path.join('output', 'evolutionary'), num_rounds=5, num_to_generate=10)
import glob
import html
import io
import os
from pathlib import Path

import cv2

import imagehash

import numpy as np

from PIL import Image

from tqdm import tqdm

def robust_imread(file_path):
    """Read an image using a more robust method for paths with special characters"""
    with open(file_path, 'rb') as f:
        img_array = np.asarray(bytearray(f.read()), dtype=np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)

def pixel_difference_count(img1, img2):
    if img1.shape != img2.shape:
        return np.inf  # Different size means definitely different
    return np.count_nonzero(img1 != img2)

def main(image_base_path, similarity_threshold=10):
    all_imgs = [os.path.splitext(x)[0] for x in glob.glob(os.path.join(image_base_path, '*.png'))]
    all_hashes = [robust_imread(x + '.png') for x in all_imgs]
    to_remove = []
    for idx in tqdm(range(len(all_hashes)), desc='Finding duplicates'):
        for jdx in range(idx + 1, len(all_hashes)):
            diff = pixel_difference_count(all_hashes[idx], all_hashes[jdx])
            if diff < similarity_threshold:
                to_remove.append(all_imgs[jdx])
    to_remove = list(set(to_remove))
    print(f'Found {len(to_remove)} duplicates')
    input('Press Enter to continue...')
    for for_removal in to_remove:
        try:
            print(f'Removing duplicate: {for_removal}')
            os.remove(for_removal + '.png')
            os.remove(for_removal + '.saml')
            all_imgs.remove(for_removal)
        except Exception as e:
            print(f'Error removing {for_removal}: {e}')

    


if __name__ == '__main__':
    main(os.path.join('output', 'base'))
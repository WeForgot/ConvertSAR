from argparse import ArgumentParser
import glob
import os
from pathlib import Path
import shutil

import cv2

import numpy as np

from tqdm import tqdm

opensae_cli_path = os.path.join('.', 'OpenSAE', 'OpenSAE.CLI.exe --input "{input_path}" --output "{output_path}"')

def dir_path(string: str) -> str:
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

def robust_imread(file_path):
    """Read an image using a more robust method for paths with special characters"""
    with open(file_path, 'rb') as f:
        img_array = np.asarray(bytearray(f.read()), dtype=np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)

def robust_imwrite(file_path, img):
    """Write an image using a more robust method for paths with special characters"""
    is_success, img_buf = cv2.imencode(Path(file_path).suffix, img)
    if is_success:
        with open(file_path, 'wb') as f:
            f.write(img_buf)
        return True
    return False
    
def center_crop(image, target_width=382, target_height=190):
    # Get the current dimensions
    height, width = image.shape[:2]
    
    # Calculate the starting coordinates
    start_x = (width - target_width) // 2
    start_y = (height - target_height) // 2
    
    # Perform the crop
    cropped_image = image[start_y:start_y+target_height, start_x:start_x+target_width]
    
    return cropped_image

def main(input_path: str, output_path: str, copy_saml: bool = False):
    all_saml = glob.glob(os.path.join(input_path, '*.saml')) + glob.glob(os.path.join(input_path, '*.sar'))
    for saml in all_saml:
        print(f'Converting {saml}')
        if saml.endswith('.saml'):
            png_path = os.path.join(output_path, os.path.basename(saml).replace('.saml', '.png'))
        elif saml.endswith('.sar'):
            png_path = os.path.join(output_path, os.path.basename(saml).replace('.sar', '.png'))
        else:
            print(f'Symbol art "{saml}" not supported.')
        print(opensae_cli_path.format(input_path=saml, output_path=png_path))
        os.system(opensae_cli_path.format(input_path=saml, output_path=png_path))
        img = robust_imread(png_path)
        img = center_crop(img)
        robust_imwrite(png_path, img)
        if copy_saml:
            shutil.copy(saml, png_path.replace('.png', '.saml'))


if __name__ == "__main__":
    parser = ArgumentParser(description="Convert SAML files to a different format.")
    parser.add_argument("--input_path", required=True, type=dir_path, help="Path to the input SAML files.")
    parser.add_argument("--output_path", type=dir_path, help="Path to save the converted files.")
    args = parser.parse_args()

    copy_saml = True

    if args.output_path is None:
        copy_saml = False
        args.output_path = args.input_path

    main(args.input_path, args.output_path, copy_saml=copy_saml)
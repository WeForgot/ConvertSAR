from math import ceil
import os
import shutil
from typing import List, Any

import cv2

import numpy as np

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

def delete_files_in_folder(folder_path):
    """Deletes all files within the specified folder.

    Args:
        folder_path: The path to the folder.
    """
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

def get_parameter_count(model: nn.Module):
    t_model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    u_model_parameters = filter(lambda p: not p.requires_grad, model.parameters())
    t_params = int(sum([np.prod(p.size()) for p in t_model_parameters]))
    u_params = int(sum([np.prod(p.size()) for p in u_model_parameters]))
    return t_params, u_params

def get_train_test_split(dataset: List[Any], train_ratio: float = 0.8):
    """Splits the dataset into training and testing sets.

    Args:
        dataset: The dataset to split.
        train_ratio: The ratio of the dataset to use for training.

    Returns:
        A tuple containing the training and testing datasets.
    """
    train_size = int(len(dataset) * train_ratio)
    train_split, test_split = dataset[:train_size], dataset[train_size:]
    return train_split, test_split

def clamp(x, min_x, max_x):
    x = min(x, max_x)
    x = max(x, min_x)
    return x


def freeze_model(model: nn.Module, freeze=True):
    for param in model.parameters():
        param.requires_grad = not freeze
    return model

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def top_p(logits, thres = 0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending = True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim = -1), dim = -1)

    sorted_indices_to_remove = cum_probs > thres
    sorted_indices_to_remove = F.pad(sorted_indices_to_remove, (1, -1), value = False)

    sorted_logits[sorted_indices_to_remove] = float('-inf')
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)

# topk

def top_k(logits, frac_num_tokens = 0.1, k = None):
    num_tokens = logits.shape[-1]

    k = default(k, ceil(frac_num_tokens * num_tokens))
    k = min(k, num_tokens)

    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

# top_a

def top_a(logits, min_p_pow = 2.0, min_p_ratio = 0.02):
    probs = F.softmax(logits, dim = -1)
    max_probs = torch.amax(probs, dim = -1, keepdim = True)
    limit = torch.pow(max_probs, min_p_pow) * min_p_ratio
    return torch.where(probs < limit, float('-inf'), logits)

def read_and_convert_img(img_path, transform):
    feature = Image.open(img_path)
    background = Image.new('RGBA', feature.size, 255)
    feature = Image.alpha_composite(background, feature).convert('RGB')
    feature = transform(feature)
    return feature

def read_raw_img(img_path):
    feature = Image.open(img_path)
    background = Image.new('RGBA', feature.size, 255)
    feature = np.array(Image.alpha_composite(background, feature).convert('RGB')).astype(np.float32)
    return feature

def read_img_cv2(img_path):
    with open(img_path, "rb") as stream:
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        feature = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED).astype(np.float32)
    return feature

def get_run_path(base_folder: str = 'runs', run_name: str = 'default', run_idx: int = None):
    if not os.path.exists(base_folder):
        os.mkdir(base_folder)
    
    if not os.path.exists(os.path.join(base_folder, run_name)):
        os.mkdir(os.path.join(base_folder, run_name))

    if run_idx is None:
        run_idx = 0

        while os.path.exists(os.path.join(base_folder, run_name, str(run_idx))):
            run_idx += 1
        run_folder = os.path.join(base_folder, run_name, str(run_idx))
        os.mkdir(run_folder)
    else:
        run_folder = os.path.join(base_folder, run_name, str(run_idx))
        if os.path.exists(run_folder):
            delete_files_in_folder(run_folder)
    
    return run_folder

def save_checkpoint(model, optimizer, epoch, path, extra=None):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    if extra:
        state.update(extra)
    torch.save(state, path)

def save_latest(run_dir, model, optimizer, epoch, extra=None):
    latest_path = os.path.join(run_dir, 'latest.pt')
    save_checkpoint(model, optimizer, epoch, latest_path, extra)

def save_best(run_dir, model, optimizer, epoch, val_loss, best_val_loss, extra=None):
    if val_loss < best_val_loss:
        best_path = os.path.join(run_dir, 'best.pt')
        save_checkpoint(model, optimizer, epoch, best_path, extra)
        return val_loss
    return best_val_loss

def load_checkpoint(path, model, optimizer=None, device='cpu'):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint.get('epoch', 0), checkpoint

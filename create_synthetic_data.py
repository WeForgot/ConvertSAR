import os
import random

import numpy as np

from tqdm import tqdm

from core.datasets.vocab import Vocabulary
from core.datasets.sa_dataset import get_data, SADataset
from core.utils.utils import read_img_cv2, convert_numpy_to_saml


def softmax_with_temp(logits, temperature=0.0, epsilon=1e-10):
    if temperature <= epsilon:  # Protect against temperature of 0
        temperature = epsilon
    scaled_logits = logits / temperature
    # Subtract max for numerical stability
    scaled_logits = scaled_logits - np.max(scaled_logits)
    exp_scaled_logits = np.exp(scaled_logits)
    sum_exp_scaled_logits = np.sum(exp_scaled_logits) + epsilon  # Add epsilon to avoid divide by zero
    probabilities = exp_scaled_logits / sum_exp_scaled_logits
    return probabilities

def random_color(red_range: tuple = (0, 255), blue_range: tuple = (0, 255), green_range: tuple = (0, 255)):
    red_val = random.randint(red_range[0], red_range[1])
    blue_val = random.randint(blue_range[0], blue_range[1])
    green_val = random.randint(green_range[0], green_range[1])
    alpha_val = random.random()
    return [red_val, blue_val, green_val, alpha_val]

def random_position(left_range: tuple = (-100, 50), right_range: tuple = (-50, 100), top_range: tuple = (-50, 100), bottom_range: tuple = (-100, 50)):
    ltx, lty = 0, 0
    lbx, lby = 0, 0
    rtx, rty = 0, 0
    rbx, rby = 0, 0
    ltx, lbx = [random.randint(left_range[0], left_range[1]) for _ in range(2)]
    rtx, rbx = [random.randint(right_range[0], right_range[1]) for _ in range(2)]
    lty, rty = [random.randint(top_range[0], top_range[1]) for _ in range(2)]
    lby, rby = [random.randint(bottom_range[0], bottom_range[1]) for _ in range(2)]
    return [ltx, lty, lbx, lby, rtx, rty, rbx, rby]

import numpy as np
from PIL import Image, ImageDraw

CANVAS_PIX = 256

def to_px(x, y):
    """Map (–127..127) → (0..CANVAS_PIX‑1) pixel coords."""
    ix = int((x + 127) / 254 * (CANVAS_PIX-1))
    iy = int((127 - y) / 254 * (CANVAS_PIX-1))
    return ix, iy

def quad_mask(coords):
    """
    coords: list of 8 floats [ltx, lty, lbx, lby, rtx, rty, rbx, rby]
    → returns a boolean mask of the filled quad.
    """
    img = Image.new("1", (CANVAS_PIX, CANVAS_PIX), 0)
    draw = ImageDraw.Draw(img)
    px_poly = [to_px(x, y) for x,y in zip(coords[0::2], coords[1::2])]
    draw.polygon(px_poly, fill=1)
    return np.array(img, dtype=bool)

VISIBLE_X = (-93,  93)
VISIBLE_Y = (-91,  91)

def is_partially_visible(coords):
    xs = coords[0::2]
    ys = coords[1::2]
    if max(xs) < VISIBLE_X[0] or min(xs) > VISIBLE_X[1]:
        return False
    if max(ys) < VISIBLE_Y[0] or min(ys) > VISIBLE_Y[1]:
        return False
    return True

def create_layers(vocab: Vocabulary, n_layers: int, max_attempts: int = 10, IOU_THRESHOLD: float = 0.7, layer_counts: np.ndarray = None):
    placed_masks = []
    layers_to_return = []

    occ = np.zeros((CANVAS_PIX, CANVAS_PIX), dtype=bool)

    if layer_counts is None:
        layer_counts = np.zeros(len(vocab) - 3, dtype=int)

    for layer_i in range(n_layers):
        colors = np.random.uniform(0.0, 1.0, size=4).tolist()
        colors[:3] = [int(x * 255) for x in colors[:3]]

        max_ct = layer_counts.max()
        scores = (max_ct + 1) - layer_counts
        probs = scores / scores.sum()
        idx = np.random.choice(len(layer_counts), p=probs)
        layer_counts[idx] += 1
        coords = None

        for attempt in range(max_attempts):
            coords = np.random.uniform(-127, 127, size=8).tolist()

            if not is_partially_visible(coords):
                continue

            mask = quad_mask(coords)

            overlap = (mask & occ).sum()
            frac = overlap / mask.sum()

            if frac < IOU_THRESHOLD:
                placed_masks.append(mask)
                occ |= mask
                break
        else:
            placed_masks.append(mask)
            occ |= mask
        
        coords = list(map(int, coords))
        layers_to_return.append([vocab[idx + 3]] + colors + coords)
    return layers_to_return, layer_counts

if __name__ == '__main__':
    vocab, data = get_data()
    num_to_create = 10000
    layer_counts = None
    for idx in tqdm(range(num_to_create), desc='SAMLs created'):
        layers_returned, layer_counts = create_layers(vocab, random.randint(1, 6), layer_counts=layer_counts)
        convert_numpy_to_saml(layers_returned, os.path.join('.','output', 'synthetic', 'basic', str(idx) + '.saml'))
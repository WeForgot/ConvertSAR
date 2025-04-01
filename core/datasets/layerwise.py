import os
import cv2
import numpy as np

# --- Updated canvas_to_pixel ---
def canvas_to_pixel(x, y):
    """
    Convert SAML canvas coordinates (x,y in [-127,127], with y positive upward originally)
    to image pixel coordinates in a 512x512 canvas.
    Here, we removed the negative on y to flip the image, so note that your cropping must be adjusted.
    """
    pixel_x = ((x + 127.0) / 254.0) * 511.0
    pixel_y = ((y + 127.0) / 254.0) * 511.0  # removed negative sign
    return [pixel_x, pixel_y]

# --- Convert a SAML layer into a warped RGBA image ---
def convert_layer(image_path, rgba_color, quad_coords):
    """
    image_path: path to the asset (shape) image.
    rgba_color: list of 4 ints [R,G,B,A] (0-255)
    quad_coords: list of four (x,y) points (in SAML canvas space) in order:
                 [top-left, top-right, bottom-right, bottom-left]
    Returns a warped layer as a 512x512 RGBA image (numpy array).
    """
    # Load asset image (with alpha) and convert to RGBA.
    asset = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if asset is None:
        raise ValueError(f"Could not load asset image from {image_path}")
    if asset.shape[2] == 4:
        asset = cv2.cvtColor(asset, cv2.COLOR_BGRA2RGBA)
    else:
        asset = cv2.cvtColor(asset, cv2.COLOR_BGR2RGBA)

    # Colorize asset: assume asset is a grayscale/white shape.
    colored = asset.astype(np.float32)
    for i in range(3):  # Adjust R, G, B channels
        colored[:, :, i] *= (rgba_color[i] / 255.0)
    # For alpha: multiply the asset’s normalized alpha by our target alpha.
    colored[:, :, 3] = (colored[:, :, 3] / 255.0) * rgba_color[3]
    colored = np.clip(colored, 0, 255).astype(np.uint8)

    # Define source points from the asset image.
    h, w = asset.shape[:2]
    pts1 = np.array([[0, 0],
                     [w, 0],
                     [w, h],
                     [0, h]], dtype=np.float32)

    # Convert quad_coords from SAML canvas to pixel coordinates.
    pts2 = np.array([canvas_to_pixel(pt[0], pt[1]) for pt in quad_coords], dtype=np.float32)

    # Compute perspective transform and warp into our 512x512 canvas.
    M = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(colored, M, (512, 512),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0, 0))
    return warped

def alpha_blend_float32(dst, src):
    """
    Alpha composite two premultiplied RGBA float32 images in [0,1].
    (This is the same as your alpha_blend but more condensed.)
    """
    src_a = src[..., 3:4]
    dst_a = dst[..., 3:4]
    out_a = src_a + dst_a * (1.0 - src_a)
    out_rgb = src[..., :3] + dst[..., :3] * (1.0 - src_a)
    return np.concatenate([out_rgb, out_a], axis=-1)

def unpremultiply_and_to_uint8(base_float32):
    """
    Convert from premultiplied alpha float32 in [0,1] to normal RGBA uint8.
    """
    alpha = base_float32[..., 3:4]
    out = np.zeros_like(base_float32)
    # Avoid dividing by zero alpha
    nonzero = (alpha > 1e-6)
    out[..., :3] = np.divide(base_float32[..., :3], alpha, out=out[..., :3], where=nonzero)
    out[..., 3] = alpha[..., 0]
    out = np.clip(np.ceil(out * 255.0), 0, 255).astype(np.uint8)
    return out

def crop_image(img):
    """
    Crop logic from your original code:
      left, top = 65, 161
      width, height = 382, 190
    """
    left, top = 65, 161
    right, bottom = left + 382, top + 190
    # Clip to image bounds
    h, w = img.shape[:2]
    top    = max(0, min(top, h))
    bottom = max(0, min(bottom, h))
    left   = max(0, min(left, w))
    right  = max(0, min(right, w))
    return img[top:bottom, left:right]

# --- Updated alpha blending routines ---
def alpha_blend(dst, src):
    """
    Alpha composite two RGBA images (both as float32 in [0,1] and assumed premultiplied).
    """
    src_a = src[..., 3:4]
    dst_a = dst[..., 3:4]
    out_a = src_a + dst_a * (1 - src_a)
    out_rgb = src[..., :3] + dst[..., :3] * (1 - src_a)
    return np.concatenate([out_rgb, out_a], axis=-1)

def blend_layers(layers):
    """
    Given a list of RGBA images (512x512, uint8) in bottom-to-top order,
    composite them using alpha blending.
    """
    base = np.zeros_like(layers[0], dtype=np.float32)
    for layer in layers:
        src = layer.astype(np.float32) / 255.0
        # Convert to premultiplied alpha.
        src[..., :3] *= src[..., 3:4]
        base = alpha_blend(base, src)
    # Convert back from premultiplied to normal alpha.
    result = np.zeros_like(base)
    alpha = base[..., 3:4]
    nonzero = (alpha > 1e-6)
    rgb = np.zeros_like(base[..., :3])
    # Use safe division: only divide where alpha is nonzero.
    np.divide(base[..., :3], alpha, out=rgb, where=nonzero)
    result[..., :3] = rgb
    result[..., 3] = alpha[..., 0]
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    return result

# --- SAML layer and image classes ---
class Layer:
    def __init__(self, layer_array):
        """
        Expects an array of 13 values:
          [type, r, g, b, a, ltx, lty, lbx, lby, rtx, rty, rbx, rby]
        """
        self.cls_id = int(layer_array[0])
        # Assume r,g,b are integers (0–255) and a is a float in [0,1].
        self.rgb = [int(layer_array[i]) for i in range(1, 4)]
        self.alpha = int(float(layer_array[4]) * 255)
        self.rgba = [self.rgb[0], self.rgb[1], self.rgb[2], self.alpha]

        # SAML eight numbers come as: ltx, lty, lbx, lby, rtx, rty, rbx, rby.
        # Reorder them to quad order: top-left, top-right, bottom-right, bottom-left.
        ltx, lty = float(layer_array[5]), float(layer_array[6])
        lbx, lby = float(layer_array[7]), float(layer_array[8])
        rtx, rty = float(layer_array[9]), float(layer_array[10])
        rbx, rby = float(layer_array[11]), float(layer_array[12])
        self.quad_coords = [
            [ltx, lty],  # top-left
            [rtx, rty],  # top-right
            [rbx, rby],  # bottom-right
            [lbx, lby]   # bottom-left
        ]

    def __repr__(self):
        return f'Layer(cls_id={self.cls_id}, rgba={self.rgba}, quad={self.quad_coords})'

    def render(self):
        asset_path = os.path.join('assets', f'{self.cls_id + 1}.png')
        return convert_layer(asset_path, self.rgba, self.quad_coords)

class SAImage:
    def __init__(self, layer_arrays):
        self.layers = [Layer(layer) for layer in layer_arrays]

    def incremental_render(self, crop=True):
        """
        Generator: yields a series of images. On the k-th yield,
        you get the composited result of the *first k layers* (from bottom to top).
        
        Because your code originally reversed the layer order for compositing,
        we do the same here but we accumulate in a loop.
        """
        # Start with an empty "base" in float32 premultiplied alpha:
        base_canvas = np.zeros((512, 512, 4), dtype=np.float32)
        
        # The bottom-most layer is the last in self.layers (due to reversed order).
        # So we iterate reversed(self.layers) so the first iteration is bottom,
        # next iteration is the next layer above that, etc.
        #reversed_layers = list(reversed(self.layers))

        for i, layer in enumerate(self.layers, start=1):
            # Render the layer as 8-bit RGBA
            layer_img = layer.render()  # shape (512, 512, 4), dtype uint8
            # Convert to float32 premultiplied
            layer_float = layer_img.astype(np.float32) / 255.0
            layer_float[..., :3] *= layer_float[..., 3:4]
            
            # Blend onto base_canvas
            base_canvas = alpha_blend_float32(base_canvas, layer_float)

            # Un-premultiply to get a normal RGBA result
            result_rgba = unpremultiply_and_to_uint8(base_canvas)
            
            # Crop if requested
            if crop:
                result_rgba = crop_image(result_rgba)

            # Yield the composited image up to this layer
            yield result_rgba

    def render(self, crop=True):
        """
        Original style: fully composite all layers (bottom→top)
        and return one final image.
        """
        # Just take the *last* yield from incremental_render
        final_image = None
        for img in self.incremental_render(crop=crop):
            final_image = img
        return final_image
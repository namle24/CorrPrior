import hashlib

import cv2
import numpy as np
import os
from skimage.segmentation import slic

def compute_complexity_mask(image, degrade_ratio=0.75, patch_size=8):
    """Select an exact fraction of patches by Sobel complexity.

    Percentile thresholding with a strict inequality can select no patches in
    a flat image and fewer than requested when many patches have the same
    score. Stable top-k selection keeps the requested support size exact and
    makes ties deterministic.
    """
    if not 0.0 < degrade_ratio <= 1.0:
        raise ValueError('degrade_ratio must be in (0, 1].')
    if patch_size <= 0:
        raise ValueError('patch_size must be positive.')
    h, w, _ = image.shape
    grid_h, grid_w = h // patch_size, w // patch_size
    if grid_h == 0 or grid_w == 0:
        raise ValueError(
            f'Image size {w}x{h} is smaller than patch size {patch_size}.'
        )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    complexity = np.sqrt(grad_x**2 + grad_y**2)
    patch_complexity = cv2.resize(complexity, (grid_w, grid_h), interpolation=cv2.INTER_AREA)
    flat_complexity = patch_complexity.reshape(-1)
    selected_count = max(
        1, min(flat_complexity.size,
               int(round(degrade_ratio * flat_complexity.size)))
    )
    # An image-dependent deterministic tie break avoids imposing the same
    # raster-order support on flat images while remaining reproducible.
    image_seed = int.from_bytes(
        hashlib.sha256(np.ascontiguousarray(image).tobytes()).digest()[:8],
        byteorder='big',
    )
    tie_break = np.random.default_rng(image_seed).random(flat_complexity.size)
    selected = np.lexsort((tie_break, -flat_complexity))[:selected_count]
    patch_mask = np.zeros(flat_complexity.size, dtype=np.float32)
    patch_mask[selected] = 1.0
    patch_mask = patch_mask.reshape(grid_h, grid_w)
    mask = cv2.resize(patch_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask

def compute_random_mask(image, degrade_ratio=0.75, patch_size=8):
    """Random patch mask."""
    h, w, _ = image.shape
    grid_h, grid_w = h // patch_size, w // patch_size
    patch_mask = np.random.rand(grid_h, grid_w) < degrade_ratio
    mask = cv2.resize(patch_mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
    return mask

def compute_slic_mask(image, degrade_ratio=0.75, n_segments=100, compactness=10):
    """SLIC superpixel based complexity mask."""
    h, w, _ = image.shape
    segments = slic(image, n_segments=n_segments, compactness=compactness, start_label=1)
    
    # Compute complexity (gradient)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    complexity = np.sqrt(grad_x**2 + grad_y**2)
    
    # Compute average complexity per segment
    segment_ids = np.unique(segments)
    seg_complexities = []
    for seg_id in segment_ids:
        seg_mask = (segments == seg_id)
        seg_complexities.append(np.mean(complexity[seg_mask]))
    
    seg_complexities = np.array(seg_complexities)
    threshold = np.percentile(seg_complexities, (1 - degrade_ratio) * 100)
    
    # Create mask based on high complexity segments
    high_complex_segs = segment_ids[seg_complexities > threshold]
    mask = np.isin(segments, high_complex_segs).astype(np.float32)
    
    return mask

if __name__ == "__main__":
    # Test script for comparison
    img_path = "Real-ESRGAN/inputs/00003.png"
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        img_vis = img.copy()
        
        mask_rand = compute_random_mask(img)
        mask_grad = compute_complexity_mask(img)
        mask_slic = compute_slic_mask(img)
        
        def apply_mask_overlay(image, mask, color):
            overlay = image.copy()
            overlay[mask == 1] = overlay[mask == 1] * 0.5 + np.array(color) * 0.5
            return overlay

        res_rand = apply_mask_overlay(img, mask_rand, [255, 0, 0]) # Blue
        res_grad = apply_mask_overlay(img, mask_grad, [0, 0, 255]) # Red
        res_slic = apply_mask_overlay(img, mask_slic, [0, 255, 0]) # Green
        
        cv2.putText(res_rand, "Random Mask", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(res_grad, "Gradient Mask", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(res_slic, "SLIC Mask", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        top = np.hstack((img, res_rand))
        bottom = np.hstack((res_grad, res_slic))
        combined = np.vstack((top, bottom))
        
        output_dir = "PD_MAE_SR/results/foundation"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "mask_comparison.png")
        cv2.imwrite(output_path, combined)
        print(f"Comparison visualization saved to {output_path}")
    else:
        print(f"File {img_path} not found.")

import cv2
import torch
import numpy as np
import torchvision.transforms as T

# Classical shadow mask (ISTD)

def classical_shadow_mask(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.medianBlur(mask, 7)
    return mask

# Classical reflection mask (SIRR)

def classical_reflection_mask(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)

    mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, bright = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    _, low_grad = cv2.threshold(mag_norm, 40, 255, cv2.THRESH_BINARY_INV)

    refl = cv2.bitwise_and(bright, low_grad)
    refl = cv2.morphologyEx(refl, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
    return refl

def classical_reflection_edges(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Convert tensor → image

def tensor_to_image(t):
    t = t.detach().cpu()
    t = (t * 0.5 + 0.5).clamp(0,1)        # [-1,1] -> [0,1]
    t = (t.permute(1,2,0).numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(t, cv2.COLOR_RGB2BGR)
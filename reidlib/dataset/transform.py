import numpy as np
import random
import cv2
def background_switch(img_a, img_b, bg_a_mask, bg_b_mask, p):
    if random.random() >= p:
        img_a = np.array(img_a)
        return img_a
    img_a = np.array(img_a)
    img_b = np.array(img_b)
    bg_a_mask = np.array(bg_a_mask) == 0
    bg_b_mask = np.array(bg_b_mask) == 0
    bg_b_mask = bg_b_mask[..., np.newaxis] * np.ones_like(img_b)
    bg_a_mask = bg_a_mask[..., np.newaxis] * np.ones_like(img_a)
    img_b = cv2.resize(img_b, img_a.shape[::-1][1:])
    bg_b_mask = cv2.resize(bg_b_mask, img_a.shape[::-1][1:], interpolation=cv2.INTER_NEAREST)

    bg_b = img_b * bg_b_mask
    img_a = img_a * (1 - bg_a_mask)
    img_s = img_a + bg_b
    return img_s
#!/usr/bin/env python
# -*- coding: utf8 -*-

import numpy as np
import pylab as plt
from PIL import Image
import matplotlib.colors as mc


def smoother(x, window):
    w = np.hanning(max(1,window))
    return np.convolve(x, w/np.sum(w), mode="same")

def find_peaks(data):
    values = np.clip(data.flatten(), 0, 1)
    bins = np.round(np.linspace(-0.1,1.1,121), 2)
    idx = 0.5 * (bins[:-1] + bins[1:])
    hist, _ = np.histogram(values, bins)
    windows = np.logspace(0,6,13,base=2)[::-1]
    peaks = np.arange(2)
    for window in windows:
        smooth = smoother(hist, window)
        smooth = np.sum(hist) * smooth / np.sum(smooth)
        diff = np.diff(smooth)
        mask = (diff[:-1] > 0) & (diff[1:] <= 0)
        if np.sum(mask) >= 2:
            imax = idx[1:-1][mask]
            err = hist > 5 * smooth
            imax = np.concatenate([imax, idx[err]])
            imax = np.unique(imax)
            i = np.argmax(np.diff(imax))
            peaks = np.array([imax[i], imax[i+1]])
            break
    return peaks


def find_mask(image, midpoint=None, grad_range=None):
    # Read image
    if isinstance(image, str):
        image = Image.open(image)
    # Convert to numpy array
    x = np.asarray(image)
    # Skip if image is already a mask
    if x.max() <= 1 + 1e-5:
        mask = x
    else:
        # Convert 255 to 1.0
        x = x.astype(float) / 255 if x.max() > 1 else x

        nx, ny = x.shape[:2]
        has_alpha = (x.ndim > 2) & (x.shape[-1] % 2 == 0)

        # Split alpha and color arrays
        alpha = x[...,-1] if has_alpha else np.ones((nx, ny))
        value = x[...,:-1] if has_alpha else x
        # Sum intensity of all color channels
        value = np.mean(value, -1) if value.ndim > 2 else value
        # Replace transparent areas with white
        value = np.stack([value, 1 - alpha])
        value = np.max(value, 0)

        # Set midpoint and color gradient range
        if midpoint is not None and grad_range is not None:
            v, dv = midpoint, grad_range
        else:
            # If not given - find peaks and set midpoint inbetween
            peaks = find_peaks(value)
            dv = 0.2 * np.diff(peaks)
            v = peaks[0] + 2 * dv

        # Calc mask (1 - black, 0 - white)
        v0, v1 = v - dv / 2, v + dv / 2
        mask = (v1 - value) / (v1 - v0)
        mask = np.clip(mask, 0, 1)
    return mask

def colorize(image, color="blue", bg="white", midpoint=None, grad_range=None):
    # Read image
    if isinstance(image, str):
        image = Image.open(image)
    # Convert to numpy array
    x = np.asarray(image)
    # Get mask (1 - black, 0 - white)
    f = find_mask(x, midpoint, grad_range)
    
    # Calc RGBA channels based on color fractions
    c0 = mc.to_rgba(color)
    c1 = mc.to_rgba(bg)
    rgba = [f * c0[i] + (1 - f) * c1[i] for i in range(4)]
    rgba = np.stack(rgba, 2)
    print(rgba.shape)
    
    # Convert RGBA to image
    rgba = 255 * np.clip(rgba, 0, 1)
    img = Image.fromarray(rgba.astype(np.uint8), "RGBA")
    return img



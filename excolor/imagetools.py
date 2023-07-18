#!/usr/bin/env python
# -*- coding: utf8 -*-

import requests
import numpy as np
import pylab as plt
from PIL import Image
import matplotlib.colors as mc


def remove_margins():
    """ 
    Removes figure margins in matplotlib to keeps only the plot area 

    """
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0,0)
    return


def load_image(fname):
    """
    Converts image to numpy array

    Parameters
    ----------
    image : str 
        Image path or url

    Returns
    -------
    img : PIL.PngImagePlugin.PngImageFile
        Image

    """
    if fname.find("http://") == 0 or fname.find("https://") == 0:
        img = Image.open(requests.get(fname, stream=True).raw)
    else:
        img = Image.open(fname)
    return img


def show_image(image, figsize=None):
    """
    Shows image without matplotlib axes and margins

    Parameters
    ----------
    image : str or PIL.PngImagePlugin.PngImageFile
        Image or image path or url

    """
    plt.figure(figsize=figsize, facecolor="#00000000")
    plt.imshow(image)
    plt.gca().set_axis_off()
    remove_margins()
    plt.show()
    return


def image_to_array(image):
    """
    Converts image to numpy array

    Parameters
    ----------
    image : str or PIL.PngImagePlugin.PngImageFile
        Image or image path or url

    Returns
    -------
    x : ndarray
        Numpy array of size (nx, ny) or (nx, ny, nchannels)

    """
    if isinstance(image, str):
        image = load_image(image)
    x = np.asarray(image)
    return x


def array_to_image(x):
    """
    Converts numpy array to image

    Parameters
    ----------
    x : ndarray
        Numpy array of size (nx, ny, nchannels), nchannels = 3 or 4

    Returns
    -------
    img : PIL.PngImagePlugin.PngImageFile
        Image

    """
    if x.max() <= 1 + 1e-5:
        x = 255 * np.clip(x, 0, 1)
    img = Image.fromarray(x.astype(np.uint8), "RGBA")
    return img
    

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


def mask_by_lightness(image, midpoint=None, grad_range=None):
    # Read image and convert to numpy array
    x = image_to_array(image)
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
    # Read image and convert to numpy array
    x = image_to_array(image)

    # Get mask (1 - black, 0 - white)
    f = mask_by_lightness(x, midpoint, grad_range)
    
    # Calc RGBA channels based on color fractions
    c0 = mc.to_rgba(color)
    c1 = mc.to_rgba(bg)
    rgba = [f * c0[i] + (1 - f) * c1[i] for i in range(4)]
    rgba = np.stack(rgba, 2)
    
    # Convert RGBA to image
    img = array_to_image(np.clip(rgba, 0, 1))
    return img



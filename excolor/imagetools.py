#!/usr/bin/env python
# -*- coding: utf8 -*-

import requests
import numpy as np
import pylab as plt
from PIL import Image
import matplotlib.colors as mc
import cv2 



def remove_margins():
    """ 
    Removes figure margins in matplotlib to keeps only the plot area 

    """
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0,0)
    plt.axis("off")
    return


def to_prime_factors(n):
    """
    Utility function to split a number into prime factors
    
    Parameters
    ----------
    n : int
        Width or height

    Returns
    -------
    factors : list
        Prime number factors

    """
    factors = []
    i = 2
    while i * i < n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    factors.append(n)
    factors.append(1)
    return factors


def size_to_size_and_dpi(size):
    """
    Splits image size (pixels) into size (~5 x 5 inches) and dpi
    
    Parameters
    ----------
    size : tuple
        Width and height [pixels]

    Returns
    -------
    size : tuple
        Output image size [inches]
    dpi : int
        Output image resolution [dots per inch]

    """
    dpi = np.gcd(*size)
    size = np.asarray(size) // dpi
    dpi = to_prime_factors(dpi)
    while size.min() < 5 and max(dpi) > 1:
        size = dpi.pop(0) * size
    dpi = np.prod(dpi)
    return size, dpi


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
    remove_margins()
    plt.show()
    return


def image_to_array(image, dpi=72):
    """
    Converts image to numpy array

    Parameters
    ----------
    image : str, ndarray, PIL.PngImagePlugin.PngImageFile or matplotlib.figure.Figure
        Figure, image or image path or url
    dpi : int, default 72
        Image resolution [dpi]; only apllies to matplotlib.figure.Figure

    Returns
    -------
    x : ndarray
        Numpy array of size (height, width) or (height, width, nchannels)
        Data type: uint8, 0-255

    """
    if isinstance(image, str):
        image = load_image(image)
    try:
        image.set_dpi(dpi)
        image.canvas.draw()
        x = np.asarray(image.canvas.renderer._renderer)
    except:
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
    """
    Performs running window smooth

    Parameters
    ----------
    x : ndarray
        1D array of data
    window : int
        Window size

    Returns
    -------
    s : ndarray
        1D array of smoothed data

    """
    w = np.hanning(max(1,window))
    s = np.convolve(x, w/np.sum(w), mode="same")
    return s


def find_peaks(data):
    """
    Finds local peaks in distribution of color intensities

    Parameters
    ----------
    data : ndarray
        1D array of color intensities

    Returns
    -------
    peaks : ndarray
        1D array of intensity peak positions

    """
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


def get_mask(image, midpoint=None, grad_range=None, dpi=72, uint8=False):
    """
    Converts image to a mask

    Parameters
    ----------
    image : str, ndarray, PIL.PngImagePlugin.PngImageFile or matplotlib.figure.Figure
        1D array of data
    midpoint : float or None, default None
        Midpoint between dark and light areas in range (0,1)
    grad_range : float or None, default None
        Gradient range between dark and light
    dpi : int, default 72
        Image resolution [dpi]; only apllies to matplotlib.figure.Figure
    uint8 : bool, default False
        Flag to cast data to np.uint8 to save disk space.

    Returns
    -------
    mask : ndarray
        float: 1.0 - object, 0.0 - background, or
        uint8: 255 - object, 0 - background

    """
    # Read image and convert to numpy array
    x = image_to_array(image, dpi)
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
        elif midpoint is not None:
            grad_range = min(midpoint, 1 - midpoint) / 2
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
    if uint8:
        mask = (255 * mask).astype(np.uint8)
    return mask


def colorize(image, color="blue", bg="white", midpoint=None, grad_range=None):
    """
    Colorizes a b&w image

    Parameters
    ----------
    image : str or PIL.PngImagePlugin.PngImageFile
        Image or image path or url
    color : str, or matplotlib.colors.Colormap, default 'blue'
        Color for object
    bg : str, or matplotlib.colors.Colormap, default 'white'
        Color for background (can be transparent '#00000000')
    midpoint : float or None, default None
        Midpoint between dark and light areas in range (0,1)
    grad_range : float or None, default None
        Gradient range between dark and light

    Returns
    -------
    img : PIL.PngImagePlugin.PngImageFile
        Colorized image

    """
    # Read image and convert to numpy array
    x = image_to_array(image)

    # Get mask (1 - black, 0 - white)
    f = get_mask(x, midpoint, grad_range)
    
    # Calc RGBA channels based on color fractions
    c0 = mc.to_rgba(color)
    c1 = mc.to_rgba(bg)
    rgba = [f * c0[i] + (1 - f) * c1[i] for i in range(4)]
    rgba = np.stack(rgba, 2)
    
    # Convert RGBA to image
    img = array_to_image(np.clip(rgba, 0, 1))
    return img


def add_shadow(fig, kernel=(31,31), sigma=10, color="#000000"):
    """
    Adds shadow to an object (assuming whate or transparent areas are background)
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Image of object on white or transparent background
    kernel : tuple, default (31,31)
        Kernel size for gaussian blur
    sigma : float, default 10
        Sigma for gaussian blur
    color : str or matplotlib.colors.Colormap, default "#000000"
        Shadow color

    Returns
    -------
    shadow : ndarray
        Image of object with shadow

    """
    fig.canvas.draw()
    img = np.asarray(fig.canvas.renderer._renderer)
    mask = get_mask(img, midpoint=0.99, grad_range=0.0, uint8=True)
    shadow = cv2.GaussianBlur(mask, kernel, sigma)
    shadow[mask>0] = 255
    channels = []
    color = 255 * np.asarray(mc.to_rgb(color))
    color = color.astype(np.uint8)
    for i in range(3):
        channel = img[...,i]
        channel[mask==0] = color[i]
        channels.append(channel)
    channels.append(shadow)
    shadow = np.stack(channels, axis=2)
    return shadow


def add_glow(fig, kernel=(31,31), sigma=10, include_alpha=False):
    """
    Adds glow to an object (assuming whate or transparent areas are background)
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Image of object on white or transparent background
    kernel : tuple, default (31,31)
        Kernel size for gaussian blur
    sigma : float, default 10
        Sigma for gaussian blur
    include_alpha : bool, default False
        Flag to include alpha layer (transparency)

    Returns
    -------
    glow : ndarray
        Image of object with glow

    """
    fig.canvas.draw()
    img = np.asarray(fig.canvas.renderer._renderer)
    mask = get_mask(img, midpoint=0.99, grad_range=0.0, uint8=True)
    img = [img[...,i] for i in range(3)] + [mask]
    channels = []
    for i in range(3 + include_alpha):
        layer = img[i]
        layer[mask==0] = 0
        blurred = cv2.GaussianBlur(layer, kernel, sigma)
        blurred = np.stack([layer, blurred])
        channel = np.max(blurred, axis=0)
        channels.append(channel)
    glow = np.stack(channels, axis=2)
    return glow



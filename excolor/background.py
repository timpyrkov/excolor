#!/usr/bin/env python
# -*- coding: utf8 -*-

import numpy as np
import pylab as plt
from PIL import Image
import matplotlib.colors as mc
from excolor.utils import _is_qualitative

import warnings
warnings.filterwarnings("ignore")


def background_color(c):
    """
    Gets color for background

    Parameters
    ----------
    c : list, str, or matplotlib.colors.Colormap object
        List of colors or a colormap

    Returns
    -------
    colors : list
        List of colors

    """
    if isinstance(c, list) or isinstance(c, np.ndarray):
        color = c
    else:
        cmap = plt.get_cmap(c)
        if cmap.name.find("gruvbox") >= 0:
            if cmap.name.find("light") >= 0:
                color = "#FBF1C7"
            else:
                color = "#1D2021"
        elif cmap.name.find("cyberpunk") >= 0:
            color = "#0D0018"
        elif cmap.name.find("synthwave") >= 0:
            color = "#0D0018"
        elif _is_qualitative(cmap):
            color = cmap.colors
            i0, i1, i2 = 0, len(color) // 2, len(color) - 1
            color = [color[i0], color[i1], color[i2]]
        else:
            color = cmap([0, 0.5, 1])
    if isinstance(color, list) or isinstance(color, np.ndarray):
        color = [mc.to_hex(c).upper() for c in color]
    return color


def background_gradient(c, fname=None, size=(1280,720)):
    """
    Gets background based on colors or cmap

    Parameters
    ----------
    c : list, str, or matplotlib.colors.Colormap object
        List of colors or a colormap
    fname : str or None, default None
        If fname given - saves image to file
    size : tuple, default (1280, 720)
        Size of output background image

    Returns
    -------
    img : PIL.PngImagePlugin.PngImageFile
        Backgraound image

    """
    epsilon = 1e-5
    colors = background_color(c)
    colors = colors if isinstance(colors, list) else [colors] * 3
    rgb = np.stack([mc.to_rgb(c) for c in colors]).T
    nx, ny = size
    x = np.arange(nx)
    y = np.arange(ny)
    x, y = np.meshgrid(x, y, indexing="xy")
    xs = np.stack([x.flatten(), y.flatten()]).T
    x0 = np.stack(set_sources(nx, ny, len(colors))).T
    r = np.stack([np.sqrt(np.sum((xs - x_)**2, 1)) for x_ in x0])
    f = np.power(r + epsilon, -1)
    f = f / np.sum(f, axis=0)
    rgba = [np.dot(rgb[i], f) for i in range(3)] + [np.ones(nx * ny)]
    rgba = np.stack([rgba[i].reshape(ny,nx) for i in range(4)], axis=2)
    rgba = np.clip(255 * rgba, 0, 255).astype(np.uint8)
    img = Image.fromarray(rgba, "RGBA")
    if fname is not None:
        if len(fname) < 5 or fname[-4:] not in [".png", ".jpg", ".svg"]:
            fname = f"{fname}.png"
        img.save(fname)
    else:
        plt.figure(facecolor="#00000000")
        plt.imshow(img)
        plt.gca().set_axis_off()
        plt.show()
    return img


def set_sources(nx, ny, n):
    """
    Sets coordinates of color sources for background gradiaents
    
    Parameters
    ----------
    nx : int
        Number of pixels along horizontal axis
    ny : init
        Number of pixels along vertical axis
    n : int
        Number of sources

    Returns
    -------
    x : ndarray
        X-coordinates of sources
    y : ndarray
        Y-coordinates of sources

    """
    r = np.sqrt(nx**2 + ny**2) / 2
    x0, y0 = nx/2, ny/2
    phi0 = 8 * np.pi / 10
    phi = phi0 + np.arange(n) * 2 * np.pi / n
    z = r * np.exp(1j * phi)
    x = z.real + x0
    y = z.imag + y0
    return x, y


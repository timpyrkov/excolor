#!/usr/bin/env python
# -*- coding: utf8 -*-

import numpy as np
import pylab as plt
from PIL import Image
import matplotlib.colors as mc
from pythonperlin import perlin
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from excolor.excolor import lighten, darken
from excolor.utils import _is_qualitative, _get_bgcolor_dict
from excolor.utils import _get_darkest, _is_arraylike
from excolor.utils import *
from excolor.geometry import *
from excolor.imagetools import *


import warnings
warnings.filterwarnings("ignore")


def get_bgcolor(cmap):
    """
    Gets background color for cmap

    Parameters
    ----------
    cmap : str or matplotlib.colors.Colormap object
        Colormap

    Returns
    -------
    color : str or tuple
        Background color in hex format

    """
    cmap = plt.get_cmap(cmap)
    bg_color_dict = _get_bgcolor_dict()
    if cmap.name in bg_color_dict:
        color = bg_color_dict[cmap.name]
    else:
        color = _get_darkest(cmap)
        color = darken(color, 0.8)
    return color


def set_color_sources(n, nx, ny, alpha=0):
    """
    Sets coordinates of color sources for background gradiaents
    
    Parameters
    ----------
    n : int
        Number of sources
    nx : int
        Number of pixels along horizontal axis
    ny : int
        Number of pixels along vertical axis
    alpha : float, default 0
        Start angle [degrees]

    Returns
    -------
    x : ndarray
        X-coordinates of sources
    y : ndarray
        Y-coordinates of sources

    """
    r = np.sqrt(nx**2 + ny**2) / 2
    phi0 = np.pi * alpha / 180.0
    phi = phi0 + np.arange(n) * 2 * np.pi / n
    z = r * np.exp(1j * phi)
    x = z.real
    y = z.imag
    return x, y


def show_color_sources(colors, nx=160, ny=90, alpha=0):
    """
    Shows color sorces positions

    Parameters
    ----------
    colors : list
        List of colors
    nx : int, default 160
        Number of pixels along horizontal axis
    ny : int, default 90
        Number of pixels along vertical axis
    alpha : float, default 0
        Start angle [degrees]

    """
    n = len(colors)
    x, y = set_color_sources(n, nx, ny, alpha)
    x += nx/2
    y += ny/2
    plt.figure(figsize=(6,5), facecolor="w")
    r = Rectangle((0,0), nx, ny, edgecolor="k", fill=False)
    plt.gca().add_patch(r)
    cmap = ListedColormap(colors, "cmap")
    plt.scatter(x, y, c=np.arange(n), cmap=cmap, s=100)
    for i in range(n):
        dx = min(nx, ny) / 25
        plt.text(x[i] + dx, y[i], i + 1)
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.show()
    return


def _get_background_palette(c):
    """
    Utility function to get colors for background

    Parameters
    ----------
    c : list, str, or matplotlib.colors.Colormap object
        List of colors or a colormap

    Returns
    -------
    colors : list
        List of colors

    """
    if _is_arraylike(c):
        colors = c
    else:
        cmap = plt.get_cmap(c)
        if _is_qualitative(cmap):
            colors = cmap.colors
            i0, i1, i2 = 0, len(colors) // 2, len(colors) - 1
            colors = [colors[i0], colors[i1], colors[i2]]
        else:
            colors = cmap([0, 0.5, 1])
    return colors


def get_stepwise_palette(cmap, size=(4,3)):
    """
    Gets step-wise colors

    Parameters
    ----------
    cmap : str or matplotlib.colors.Colormap object
        Colormap
    size : tuple, default (4,3)
        Number of colors in each step

    Returns
    -------
    colors : list
        List of colors

        """
    def _rank_by_value(colors):
        hsv = [mc.rgb_to_hsv(mc.to_rgb(c)) for c in colors]
        v = np.stack(hsv)[:,2]
        idx = np.arange(len(v))
        p = np.polyfit(idx[:-1], v[:-1], 1)
        if p[0] < 0:
            colors = colors[::-1]
        return colors
    n = sum(size)
    cmap = plt.get_cmap(cmap)
    colors = get_colors(cmap, n, exclude_extreme=False)
    if len(colors) != n:
        cmap = LinearSegmentedColormap.from_list("cmap", colors)
        colors = cmap(np.linspace(0, 1, n))
        colors = [mc.to_hex(c) for c in colors]
    m, n = size
    colors = _rank_by_value(colors[:m]) + _rank_by_value(colors[m:])
    hsv = [mc.rgb_to_hsv(mc.to_rgb(c)) for c in colors]
    v = np.round(np.stack(hsv)[:,2], 2)
    return colors


def background_gradient(c, size=(1280,720), alpha=0, show=None, fname=None):
    """
    Draws background based on colors or cmap

    Parameters
    ----------
    c : list, str, or matplotlib.colors.Colormap object
        List of colors or a colormap
    fname : str or None, default None
        If fname given - saves image to file
    size : tuple, default (1280, 720)
        Size of output background image [pixels]
    monochrome : bool or None, default None
        If False - use all colors, else - only the darkest color;
        If None - autoselect.

    Returns
    -------
    img : PIL.PngImagePlugin.PngImageFile
        Backgraound image

    """
    epsilon = 1e-5
    # If only one color is given, expand it to darker and lighter
    if _is_arraylike(c):
        colors = c
    else:
        try:
            color = get_bgcolor(c)
        except:
            color = c
        colors = [lighten(color), color, darken(color, 0.7), darken(color, 0.7)]
    nx, ny = size
    n = len(colors)
    # Set color source coordinates
    x0, y0 = set_color_sources(n, nx, ny, alpha)
    a0 = np.arctan2(y0, x0)
    r0 = np.sqrt(x0**2 + y0**2)
    # Set pixel coordinates
    x = np.arange(nx) - (nx - 1) / 2
    y = np.arange(ny) - (ny - 1) / 2
    x, y = np.meshgrid(x, y, indexing="xy")
    x, y = x.flatten(), y.flatten()
    a = np.arctan2(y, x)
    r = np.sqrt(x**2 + y**2)
    # Iterate through color sources
    dist = []
    for i in range(n):
        a_ = a - a0[i]
        r_ = r * np.cos(a_)
        d = r0[i] - r_
        dist.append(d)
    d = np.stack(dist)
    f = np.power(d + epsilon, -1.2)
    f = f / np.sum(f, axis=0)
    # Calc color fractions
    rgb = np.stack([mc.to_rgb(color) for color in colors]).T
    rgba = [np.dot(rgb[i], f) for i in range(3)] + [np.ones(nx * ny)]
    rgba = np.stack([rgba[i].reshape(ny,nx) for i in range(4)], axis=2)
    rgba = np.clip(255 * rgba, 0, 255).astype(np.uint8)
    img = Image.fromarray(rgba, "RGBA")
    if fname is not None:
        if len(fname) < 5 or fname[-4:] not in [".png", ".jpg", ".svg"]:
            fname = f"{fname}.png"
        img.save(fname)
    if show is None and fname is None:
        show = True
    if show == True:
        plt.figure(facecolor="#00000000")
        plt.imshow(img)
        plt.gca().set_axis_off()
        plt.show()
    plt.close()
    return img


def background_concentric_lines(colors, background, fname=None, size=(1280,720), dpi=80, center=(0,0), seed=0, nrep=4):
    """
    Draws background with distorted concentric lines

    Parameters
    ----------
    colors : list
        List of colors
    background : str, matplotlib.colors.Color or image
        Background color or image
    fname : str or None, default None
        If fname given - saves image to file
    size : tuple, default (1280,720)
        Size of output background image [pixels]
    dpi: int, default 80
        Resolution [dpi]
    center : tuple, default (0,0)
        Coordinates of circle center from the right bottom corner [inches]
    seed : int, default 0
        Random seed for generation of distorting Perlin noise
    nreps : int, default 4
        Number of repeats for colorcycle

    Returns
    -------
    fig : matplotlib.figure.Figure
        Backgraound image

    """
    n_major = nrep
    n_minor = len(colors)
    x_max = size[0] + dpi * center[0]
    y_max = size[1] - dpi * center[1]
    r_max = np.sqrt(x_max**2 + y_max**2)
    r_min = 0.6 * r_max
    dr_major = (r_max - r_min) / n_major
    dr_minor = dr_major / (n_minor + 1)
    dens = 20
    n = np.ceil(n_major * n_minor / dens).astype(int) + 1
    p = perlin((n,36), dens=dens, seed=seed)[dens//2:]
    inch_size = (size[0] / dpi, size[1] / dpi)
    try:
        bgcolor = mc.to_rgb(background)
        fig = plt.figure(figsize=inch_size, facecolor=bgcolor)
    except:
        fig = plt.figure(figsize=inch_size, facecolor="#00000000")
        img = image_to_array(background)
        plt.imshow(img[::-1])
    fig.set_dpi(dpi)
    for i in range(n_major):
        for j in range(n_minor):
            k = n_minor * i + j
            r = r_min + dr_major * i + dr_minor * j
            x, y = get_circle_dots(r, n=720)
            x, y = distort_radius(x, y, 2 * dpi * p[k])
            x += size[0] + center[0]
            y += center[1]
            lw = n_minor + 2 - 1 * (j % n_minor)
            plt.plot(x, y, lw=lw, color = colors[j])
    plt.xlim(0, size[0])
    plt.ylim(0, size[1])
    remove_margins()
    if fname is not None:
        if len(fname) < 5 or fname[-4:] not in [".png", ".jpg", ".svg"]:
            fname = f"{fname}.png"
        plt.savefig(fname, dpi=dpi)
    else:
        plt.show()
    return fig


    

def background_concentric_patches(colors, background, fname=None, size=(1280,720), dpi=80, center=(0,0), seed=0):
    """
    Draws background with distorted concentric patches

    Parameters
    ----------
    colors : list
        List of colors
    background : str, matplotlib.colors.Color or image
        Background color or image
    fname : str or None, default None
        If fname given - saves image to file
    size : tuple, default (1280,720)
        Size of output background image [pixels]
    dpi: int, default 80
        Resolution [dpi]
    center : tuple, default (0,0)
        Coordinates of circle center from the right bottom corner [inches]
    seed : int, default 0
        Random seed for generation of distorting Perlin noise

    Returns
    -------
    fig : matplotlib.figure.Figure
        Backgraound image

    """
    def _draw_patch(x, y, size, color, dpi):
        inch_size = (size[0] / dpi, size[1] / dpi)
        dx = np.array([size[0] + dpi, -dpi, -dpi, size[0] + dpi])
        dy = np.array([-dpi, -dpi, size[1] + dpi, size[1] + dpi])
        x = np.concatenate([x, dx])
        y = np.concatenate([y, dy])
        fig = plt.figure(figsize=inch_size, facecolor="#00000000")
        fig.set_dpi(dpi)
        plt.fill(x, y, c=color)
        plt.xlim(0, size[0])
        plt.ylim(0, size[1])
        remove_margins()
        x = add_shadow(fig, kernel=(301,301), sigma=100, color="#000000")
        plt.close()
        return x
    n_major = len(colors)
    x_max = size[0] + dpi * center[0]
    y_max = size[1] - dpi * center[1]
    r_max = np.sqrt(x_max**2 + y_max**2)
    r_min = 0.6 * r_max
    dr_major = (r_max - r_min) / n_major
    dens = 20
    n = np.ceil(n_major * 6 / dens).astype(int) + 1
    p = perlin((n,36), dens=dens, seed=seed)[dens//2:]
    inch_size = (size[0] / dpi, size[1] / dpi)
    layers = []
    for i in range(n_major):
        k = 8 * i
        r = r_min + dr_major * i
        x, y = get_circle_dots(r, n=720)
        x, y = distort_radius(x, y, 2 * dpi * p[k])
        x += size[0] + center[0]
        y += center[1]
        x = _draw_patch(x, y, size, colors[i], dpi)
        layers.append(x)
    try:
        bgcolor = mc.to_rgb(background)
        fig = plt.figure(figsize=inch_size, facecolor=bgcolor)
    except:
        fig = plt.figure(figsize=inch_size, facecolor="#00000000")
        img = image_to_array(background)
        plt.imshow(img[::-1])
    fig.set_dpi(dpi)
    for layer in layers:
        plt.imshow(layer[::-1])
    plt.xlim(0, size[0])
    plt.ylim(0, size[1])
    remove_margins()
    if fname is not None:
        if len(fname) < 5 or fname[-4:] not in [".png", ".jpg", ".svg"]:
            fname = f"{fname}.png"
        plt.savefig(fname, dpi=dpi)
    else:
        plt.show()
    return


import types
__all__ = [name for name, thing in globals().items()
          if not (name.startswith("_") or isinstance(thing, types.ModuleType))]
del types

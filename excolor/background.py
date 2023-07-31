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


def set_circular_sources(nx, ny, n):
    """
    Sets coordinates of color sources for background gradiaents
    
    Parameters
    ----------
    nx : int
        Number of pixels along horizontal axis
    ny : int
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
    phi0 = (0.55 + 1 / n) * np.pi
    phi = phi0 + np.arange(n) * 2 * np.pi / n
    z = r * np.exp(1j * phi)
    x = z.real + x0
    y = z.imag + y0
    return x, y


def show_circular_sources(nx=160, ny=90, n=3, c="coolwarm"):
    """
    Shows color sorces positions

    Parameters
    ----------
    nx : int, default 160
        Number of pixels along horizontal axis
    ny : int, default 90
        Number of pixels along vertical axis
    n : int, default 3
        Number of sources
    c : list, str or matplotlib.colors.Colormap object
        List of colors or a colormap

    """
    x, y = set_circular_sources(nx, ny, n)
    plt.figure(figsize=(6,5), facecolor="w")
    r = Rectangle((0,0), nx, ny, edgecolor="k", fill=False)
    plt.gca().add_patch(r)
    if isinstance(c, list):
        c = ListedColormap(c, "cmap")
    plt.scatter(x, y, c=np.arange(n), cmap=c, s=100)
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


def background_gradient(c, fname=None, size=(1280,720), monochrome=None):
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
    # If monochrome flag not given - autoselect
    c = c[0] if _is_arraylike(c) and len(c) == 1 else c
    if monochrome is None:
        try:
            cmap = plt.get_cmap(c)
            bg_color_dict = _get_bgcolor_dict()
            monochrome = cmap.name in bg_color_dict
        except:
            monochrome = not _is_arraylike(c)
    # Generate 36 colors based on monochrome or color palette
    if monochrome:
        try:
            color = get_bgcolor(c)
        except:
            color = mc.to_hex(mc.to_rgb(c))
        colors = [darken(color), darken(color), color, lighten(color)]
    else:
        colors = _get_background_palette(c)
    if len(colors) < 36:
        cmap = LinearSegmentedColormap.from_list("cmap", colors)
        colors = cmap(np.linspace(0,1,36))
    # Generate gradient pixels
    rgb = np.stack([mc.to_rgb(color) for color in colors]).T
    nx, ny = size
    x = np.arange(nx)
    y = np.arange(ny)
    x, y = np.meshgrid(x, y, indexing="xy")
    xs = np.stack([x.flatten(), y.flatten()]).T
    x0 = np.stack(set_circular_sources(nx, ny, len(colors))).T
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

#!/usr/bin/env python
# -*- coding: utf8 -*-

import numpy as np
import pylab as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.colors as mc
from cycler import cycler
from excolor.utils import _is_arraylike, _is_int, _is_exp, _is_log, _is_cmap
from excolor.utils import _is_qualitative, _is_cyclic, _is_divergent, _get_bgcolor_dict
from excolor.utils import *


import warnings
warnings.filterwarnings("ignore")


def logscale_cmap(cmap, norders=3):
    """
    Extends list of colors by adding colors inbetween

    Parameters
    ----------
    cmap : str or matplotlib.colors.Colormap object
        Colormap
    norders : int, optional
        Number of orders for logspace gradient

    Returns
    -------
    cmod : matplotlib.colors.Colormap object
        Modified colormap

    """
    cmap = plt.get_cmap(cmap)
    name = f"log_{cmap.name}"
    gradient = np.logspace(0, -norders, 2 * norders)
    colors = cmap(gradient)
    cmod = LinearSegmentedColormap.from_list(name, colors)
    return cmod


def list_cmaps(show=False):
    """
    List (and show) all registered colormaps

    Parameters
    ----------
    show : bool, default False
        Flag to show colors of cmaps

    """
    for cmap in plt.colormaps():
        labels = ['Discrete'] if _is_qualitative(cmap) else ['Continuous']
        if _is_divergent(cmap):
            labels.append('Divergent')
        if _is_cyclic(cmap):
            labels.append('Cyclic')
        labels = [f'{l:10}' for l in labels]
        prtstr = f'{cmap:20}' + (' ').join(labels)
        print(prtstr)
        if show:
            show_colors(cmap, verbose=False)
            plt.show()
    return


def show_cmap(cmap, gradient=None):
    """
    Plots colormap as colors

    Parameters
    ----------
    cmap : str or matplotlib.colors.Colormap object
        Colormap
    gradient : array-like, optional
        Linspace or logspace gradient

    """
    cmap = plt.get_cmap(cmap)
    if gradient is None:
        colors = get_colors(cmap, exclude_extreme=False)
    else:
        gradient = gradient / np.max(gradient)
        colors = plt.get_cmap(cmap)(gradient)
    colors = [color_to_hex(c) for c in colors]
    show_colors(colors, cmap.name)
    return


def show_cbar(cmap):
    """
    Plots colormap as colorbar

    Parameters
    ----------
    cmap : str or matplotlib.colors.Colormap object
        Colormap

    """
    cmap = plt.get_cmap(cmap)
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))
    plt.figure(figsize=(12,2), facecolor="#00000000")
    plt.title(cmap.name, fontsize=20, color="grey", pad=16)
    plt.imshow(gradient, aspect="auto", cmap=cmap)
    plt.xticks([0, 127, 255], [0, 0.5, 1], fontsize=16, color="grey")
    plt.yticks([])
    for e in ["top", "bottom", "right", "left"]:
        plt.gca().spines[e].set_color("#00000000")
    plt.tight_layout()
    return


def show_colors(c, title="", cname=None, verbose=True):
    """
    Plots colors from list or a colormap

    Parameters
    ----------
    c : list, str, or matplotlib.colors.Colormap object
        List of colors or a colormap
    title : str, default ''
        Figure title
    cname : list or None, default None
        List of color names
    verbose : bool, default True
        Flag to print color names

    """
    try:
        c = plt.get_cmap(c)
        colors = get_colors(c, exclude_extreme=False)
        title = c.name
    except:
        colors = c if _is_arraylike(c) else [c]
    if verbose:
        print(colors)
    d = 0.05
    width = 1  -2 * d
    n, m = aspect_ratio(len(colors), lmin=12)
    plt.figure(figsize=(2*n+4,2*m), facecolor="#00000000")
    plt.title(title, fontsize=20, color="grey")
    for k, color in enumerate(colors):
        i = k % n
        j = k // n
        rgb = color_to_rgb(color)
        if rgb is not None:
            r = Rectangle((i, -j), width, -width, facecolor=rgb, fill=True)
            plt.gca().add_patch(r)
            h, s, v = mc.rgb_to_hsv(rgb)
            fontcolor = "white" if v < 0.6 else "black"
            x, y = i + 0.5 - 2 * d, -j - 0.5
            name = cname[k] if cname is not None else color_to_hex(rgb)
            plt.text(x, y, name, fontsize=20, color=fontcolor, ha="center")
    plt.xlim(-d, n - d)
    plt.ylim(-m + d, d)
    plt.gca().set_axis_off()
    plt.tight_layout()
    return


def set_color_cycler(c, n=3):
    """
    Sets ax color cycler based on cmap or list of colors

    Parameters
    ----------
    c : list, str, or matplotlib.colors.Colormap object
        List of colors or a colormap
    n : int, optional
        Number of colors in cycler

    """
    try:
        c = plt.get_cmap(c)
        colors = get_colors(c, n, exclude_extreme=True)
    except:
        colors = c if _is_arraylike(c) else [c]
    plt.gca().set_prop_cycle(cycler(color=colors))
    return


def lighten(c, scale=0.5, keephue=True):
    """
    Lightens colors or colormap

    Parameters
    ----------
    c : list, str, or matplotlib.colors.Colormap object
        List of colors or a colormap
    scale : float, optional
        Scale in range (0,1) - propensity of lightening
    keephue : bool, default True
        If True - preserve hue

    Returns
    -------
    cmod : list or matplotlib.colors.Colormap object
        List of modified colors or colormap

    """
    try:
        c = plt.get_cmap(c)
        colors = get_colors(c, 256, exclude_extreme=False)
    except:
        colors = c if _is_arraylike(c) else [c]
    if keephue:
        hsv = np.array([mc.rgb_to_hsv(mc.to_rgb(color)) for color in colors]).T
        hsv[2] = hsv[2] + scale * (1 - hsv[2])
        cmod = [mc.to_hex(mc.hsv_to_rgb(color)).upper() for color in hsv.T]
    else:
        rgb = np.array([mc.to_rgb(color) for color in colors])
        rgb = rgb + scale * (1 - rgb)
        cmod = [mc.to_hex(color).upper() for color in rgb]
    if _is_cmap(c):
        name = c.name + "_light"
        cmod = LinearSegmentedColormap.from_list(name, cmod)
    elif len(cmod) == 1:
        cmod = cmod[0]
    return cmod


def darken(c, scale=0.5):
    """
    Darkens colors or colormap

    Parameters
    ----------
    c : list, str, or matplotlib.colors.Colormap object
        List of colors or a colormap
    scale : float, optional
        Scale in range (0,1) - propensity of darkening

    Returns
    -------
    cmod : list or matplotlib.colors.Colormap object
        List of modified colors or colormap

    """
    try:
        c = plt.get_cmap(c)
        colors = get_colors(c, 256, exclude_extreme=False)
    except:
        colors = c if _is_arraylike(c) else [c]
    hsv = np.array([mc.rgb_to_hsv(mc.to_rgb(color)) for color in colors]).T
    hsv[2] = (1 - scale) * hsv[2]
    cmod = [mc.to_hex(mc.hsv_to_rgb(color)).upper() for color in hsv.T]
    if _is_cmap(c):
        name = c.name + "_dark"
        cmod = LinearSegmentedColormap.from_list(name, cmod)
    elif len(cmod) == 1:
        cmod = cmod[0]
    return cmod


def saturate(c, scale=0.5):
    """
    Saturates colors or colormap

    Parameters
    ----------
    c : list, str, or matplotlib.colors.Colormap object
        List of colors or a colormap
    scale : float, optional
        Scale in range (0,1) - propensity of saturation

    Returns
    -------
    cmod : list or matplotlib.colors.Colormap object
        List of modified colors or colormap

    """
    try:
        c = plt.get_cmap(c)
        colors = get_colors(c, 256, exclude_extreme=False)
    except:
        colors = c if _is_arraylike(c) else [c]
    hsv = np.array([mc.rgb_to_hsv(mc.to_rgb(color)) for color in colors]).T
    hsv[1] = hsv[1] + scale * (1 - hsv[1])
    cmod = [mc.to_hex(mc.hsv_to_rgb(color)).upper() for color in hsv.T]
    if _is_cmap(c):
        name = c.name + "_saturated"
        cmod = LinearSegmentedColormap.from_list(name, cmod)
    elif len(cmod) == 1:
        cmod = cmod[0]
    return cmod


def desaturate(c, scale=0.5):
    """
    Desaturates colors or colormap

    Parameters
    ----------
    c : list, str, or matplotlib.colors.Colormap object
        List of colors or a colormap
    scale : float, optional
        Scale in range (0,1) - propensity of desaturation

    Returns
    -------
    cmod : list or matplotlib.colors.Colormap object
        List of modified colors or colormap

    """
    try:
        c = plt.get_cmap(c)
        colors = get_colors(c, 256, exclude_extreme=False)
    except:
        colors = c if _is_arraylike(c) else [c]
    hsv = np.array([mc.rgb_to_hsv(mc.to_rgb(color)) for color in colors]).T
    hsv[1] = (1 - scale) * hsv[1]
    cmod = [mc.to_hex(mc.hsv_to_rgb(color)).upper() for color in hsv.T]
    if _is_cmap(c):
        name = c.name + "_desaturated"
        cmod = LinearSegmentedColormap.from_list(name, cmod)
    elif len(cmod) == 1:
        cmod = cmod[0]
    return cmod


def color_to_hex(c, keep_alpha=False):
    """
    Converts color(s) to hex names

    Parameters
    ----------
    c : list or matplotlib.colors.Color object
        Color or list of colors
    keep_alpha : bool, default False
        Whether to keep alpha channel (opacity)

    Returns
    -------
    hexname : list or str
        Name of color or list of colors

    """
    hexname = mc.to_hex(c, keep_alpha).upper()
    return hexname


def color_to_rgb(c):
    """
    Converts color(s) to rgb

    Parameters
    ----------
    c : str, list, tuple, or matplotlib.colors.Color object
        Color

    Returns
    -------
    rgb : tuple
        RGB intensities in range [0,1]

    """
    if c is None:
        rgb = None
    else:
        try:
            # default matplotlib func
            rgb = mc.to_rgb(c)
        except:
            try:
                # rgb-int to hex then to rgb
                hcolor = "#" + "".join(["{:02X}".format(c_) for c_ in c])
                rgb = mc.to_rgb(hcolor)
            except:
                rgb = None
    return rgb


def grey_to_hue(scale=0.5):
    """
    Generates colors of different huw with a given greyscale level

    Parameters
    ----------
    scale : float, optional
        Scale in range (0,1) - level of greyscale

    Returns
    -------
    colors : list
       List of colors

    """
    n = 256
    r = np.arange(n) / (n - 1)
    g = np.arange(n) / (n - 1)
    r, g = np.meshgrid(r, g, indexing="ij")
    b = (scale - 0.299 * r - 0.587 * g) / 0.114
    rgb = np.stack([r,g,b]).reshape(3,-1).T
    rgb = rgb[(np.min(rgb, 1) >= 0) & (np.max(rgb, 1) <= 1)]
    hsv = np.stack([mc.rgb_to_hsv(rgb_) for rgb_ in rgb])
    hsv = hsv[np.argsort(hsv[:,0])]
    s = hsv[:,1]
    s_avg, s_std = np.median(s), .5 * np.std(s)
    s_min, s_max = s_avg - s_std, s_avg + s_std
    mask = (s >= s_min) & (s <= s_max)
    hsv = hsv[mask]
    colors = [mc.hsv_to_rgb(hsv_) for hsv_ in hsv]
    return colors


def grey_to_aquamarine(scale=0.5):
    """
    Generates aquamarine colors with a given greyscale level

    Parameters
    ----------
    scale : float, optional
        Scale in range (0,1) - level of greyscale

    Returns
    -------
    colors : list
       List of colors

    """
    color = grey_to_hue(scale)
    hsvs = np.array([mc.rgb_to_hsv(mc.to_rgb(c)) for c in color]).T
    colors = []
    for h in np.array([250, 210, 170]) / 360:
        d = np.abs(hsvs[0] - h)
        i = np.argmin(d)
        colors.append(color[i])
    return colors


def aspect_ratio(length, lmin=0):
    """
    Finds optimal aspect ratio to represent sequence of charts

    Parameters
    ----------
    length : int
        Number of items
    lmin : int, optional
        Min number of items to start splitting into rows

    Returns
    -------
    n : int
       Number of columns
    m : int
        Number of rows

    """
    if length > 0:
        d = np.array([-2, -1, 0, 1, 2])
        n0 = np.sqrt(length / 2)
        ns = []
        ms = []
        ds = []
        for s in [4, 5, 6]:
            n1 = (2 * n0 // s + d).astype(int) * s
            m1 = np.ceil(length / n1).astype(int)
            for k in range(len(n1)):
                if n1[k] > 0 and m1[k] > 0 and n1[k] > m1[k]:
                    delta = n1[k] * m1[k] - length
                    if delta >= 0:
                        ns.append(n1[k])
                        ms.append(m1[k])
                        ds.append(delta)
        mask = np.array(ds) == min(ds)
        ns = np.array(ns)[mask]
        ms = np.array(ms)[mask]
        idx = np.argmin(np.abs(ns / ms - 5))
        n, m = ns[idx], ms[idx]
        if isinstance(n, np.ndarray):
            n, m = n[0], m[0]
        if length <= lmin or n > length:
            n = length
    else:
        n, m = 0, 0
    return n, m



""" Aliases for functions """
# show_colormap = show_cmap
# show_colorbar = show_cbar
gray_to_hue = grey_to_hue
gray_to_aquamarine = grey_to_aquamarine

import types
__all__ = [name for name, thing in globals().items()
          if not (name.startswith("_") or isinstance(thing, types.ModuleType))]
del types


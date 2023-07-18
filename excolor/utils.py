#!/usr/bin/env python
# -*- coding: utf8 -*-

import numpy as np
import pylab as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.colors as mc


def _is_arraylike(x, epsilon=1e-5):
    """
    Tests if argument is array-like

    """
    mask = isinstance(x, np.ndarray) or isinstance(x, list)
    mask = mask or isinstance(x, tuple) or isinstance(x, set)
    return mask


def _is_int(x, epsilon=1e-5):
    """
    Tests if argument is integer

    """
    if isinstance(x, np.ndarray) or isinstance(x, list):
        mask = np.all([is_int(x_) for x_ in x])
    else:
        mask = np.abs(x - np.round(x)) < epsilon
    return mask


def _is_exp(x):
    """
    Tests if number are exponentially scaled

    """
    mask = np.isfinite(x)
    t = np.arange(len(x))
    err = []
    for x_ in [x, np.log(x)]:
        mask = np.isfinite(x_)
        if np.any(mask):
            p = np.polyfit(t[mask], x_[mask], 1)
            x_pred = np.polyval(p, t)
            dx = (x_ - x_pred)**2
            dx = np.sqrt(np.nanmean(dx))
        else:
            dx = np.inf
        err.append(dx)
    mask = err[1] < err[0]
    return mask


def _is_log(x):
    """
    Tests if number are logarithmically scaled

    """
    mask = np.isfinite(x)
    t = np.arange(len(x))
    err = []
    for x_ in [x, np.exp(x)]:
        mask = np.isfinite(x_)
        if np.any(mask):
            p = np.polyfit(t[mask], x_[mask], 1)
            x_pred = np.polyval(p, t)
            dx = (x_ - x_pred)**2
            dx = np.sqrt(np.nanmean(dx))
        else:
            dx = np.inf
        err.append(dx)
    mask = err[1] < err[0]
    return mask


def _is_qualitative(cmap):
    """
    Tests if colormap is qualitative (categorical)

    """
    cmap = plt.get_cmap(cmap)
    try:
        colors = cmap.colors
        mask = len(colors) <= 20
    except:
        mask = False
    return mask


def _is_cyclic(cmap):
    """
    Tests if colormap is cyclic 
    - i.e. Hue, Saturation, and Value are the same at both ends

    """
    n = 32
    colors = get_colors(cmap, n, exclude_extreme=False)
    hsv = np.array([mc.rgb_to_hsv(mc.to_rgb(color)) for color in colors])
    d = np.abs(hsv[-1] - hsv[0])
    d[0] = min(d[0], abs(d[0] - 1), abs(d[0] + 1))
    mask = np.all(d < 0.1) & (~_is_qualitative(cmap))
    return mask



def _is_divergent(cmap):
    """
    Tests if colormap is cyclic - i.e. middle color is desaturated

    """
    def _fix_hue_phase(h):
        h_ext = np.stack([h - 1, h, h + 1]).T
        h_fix = [h[0]]
        for i in range(len(h) - 1):
            d = np.abs(h_ext[i+1] - h_fix[i])
            j = np.argmin(d)
            h_fix.append(h_ext[i+1][j])
        h_fix = np.array(h_fix)
        return h_fix

    n = 32
    colors = get_colors(cmap, n, exclude_extreme=False)
    h, s, v = np.array([mc.rgb_to_hsv(mc.to_rgb(color)) for color in colors]).T
    d = np.abs(np.diff(_fix_hue_phase(h)))
    i = np.argmax(d)
    hmask = (d[i] > 0.05) & (abs(i - n / 2) < 4)
    
    p = np.polyfit(np.arange(len(v)), v, 2)
    i = -p[1] / (2 * p[0]) if p[0] != 0 else 0
    vmask = (abs(p[0]) > 0.001) & (abs(i - n / 2) < 4)

    p = np.polyfit(np.arange(len(s)), s, 2)
    i = -p[1] / (2 * p[0]) if p[0] != 0 else 0
    smask = (abs(p[0]) > 0.001) & (abs(i - n / 2) < 3)
    
    mask = np.sum([hmask, smask, vmask]) >= 2
    mask = mask & (~_is_cyclic(cmap)) & (~_is_qualitative(cmap))
    return mask



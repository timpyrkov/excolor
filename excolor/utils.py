#!/usr/bin/env python
# -*- coding: utf8 -*-

import numpy as np
import pylab as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.colors as mc


def extend_colors(c, n=5):
    """
    Extends list of colors by adding colors inbetween

    Parameters
    ----------
    c : list
        List of colors
    n : int, optional
        Number of colors to output

    Returns
    -------
    colors : list
       List of colors

    """
    gradient = np.linspace(0,1,n)
    cmap = LinearSegmentedColormap.from_list("cmap", c)
    colors = cmap(gradient)
    colors = [mc.to_hex(color).upper() for color in colors]
    return colors


def get_colors(cmap, n=None, exclude_extreme=True):
    """
    Gets colors from cmap

    Parameters
    ----------
    cmap : str or matplotlib.colors.Colormap object
        Colormap
    n : int, optional
        Number of colors
    exclude_extreme : bool, default True
        Whethr to exclude extreme (very dark/light) colors

    Returns
    -------
    colors : list
        List of colors in hex format

    """
    cmap = plt.get_cmap(cmap)
    if _is_qualitative(cmap):
        colors = cmap.colors
        if n is not None:
            colors = extend_colors(colors, n=10*len(colors))
            cmap = LinearSegmentedColormap.from_list("cmap", colors)
            colors = get_colors(cmap, n, exclude_extreme=False)
    else:
        n = 10 - _is_divergent(cmap) if n is None else n
        dn = 1 if exclude_extreme else 0
        idx = np.arange(dn, n + dn) / (n + 2 * dn - 1)
        colors = cmap(idx)
    colors = [mc.to_hex(c, keep_alpha=False).upper() for c in colors]
    return colors



def _is_arraylike(x):
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


def _is_cmap(c):
    """
    Tests if the argument is cmap name or matplotlib.colors.Colormap object

    """
    try:
        c = plt.get_cmap(c)
        is_cmap = True
    except:
        is_cmap = False
    return is_cmap



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


def _register_cmap(cmap):
    """
    Registers matplotlib colormap in the current session

    """
    try:
        plt.colormaps.register(cmap)
    except:
        pass


def _add_cool_warm_colormaps():
    """
    Extends list of registered colormaps by modifications of 'coolwarm'

    """
    # Divergent to sequential
    gradient = np.linspace(0.5, 1, 128)
    dct = {"cool": "cold", "warm": "warm"}
    warm_colors = plt.get_cmap("coolwarm")(gradient)
    cool_colors = plt.get_cmap("coolwarm_r")(gradient)
    cmap = LinearSegmentedColormap.from_list("warm", warm_colors)
    _register_cmap(cmap)
    cmap = LinearSegmentedColormap.from_list("warm_r", warm_colors[::-1])
    _register_cmap(cmap)
    cmap = LinearSegmentedColormap.from_list("cold", cool_colors)
    _register_cmap(cmap)
    cmap = LinearSegmentedColormap.from_list("cold_r", cool_colors[::-1])
    _register_cmap(cmap)
    # Log-scaled
    gradient = np.logspace(0, -10, 128)
    logwarm_colors = plt.get_cmap("warm")(gradient)
    logwarm_r_colors = plt.get_cmap("warm_r")(gradient)
    cmap = LinearSegmentedColormap.from_list("logwarm", logwarm_colors)
    _register_cmap(cmap)
    cmap = LinearSegmentedColormap.from_list("logwarm_r", logwarm_r_colors)
    _register_cmap(cmap)
    logcool_colors = plt.get_cmap("cold")(gradient)
    logcool_r_colors = plt.get_cmap("cold_r")(gradient)
    cmap = LinearSegmentedColormap.from_list("logcool", logcool_colors)
    _register_cmap(cmap)
    cmap = LinearSegmentedColormap.from_list("logcool_r", logcool_r_colors)
    _register_cmap(cmap)
    return


def _add_extended_colormaps():
    """
    Extends list of registered colormaps by modifications of 'coolwarm'
    and manually hardcoded colormaps

    """
    try:
        add_cool_warm_colormaps()
    except:
        pass
    # aquamarine = grey_to_aquamarine()
    # aquamarine_light = lighten(aquamarine, 0.8)
    # aquamarine_dark = darken(aquamarine, 0.4)
    color_dict = {
        "BrBu": ["#9B2227", "#BA3F04", "#CA6705", "#EE9B04", "#EAD7A4", "#93D3BD", "#4BB3A9", "#039396", "#027984", "#015F72"],
        "BrGn": ["#9B2227", "#BA3F04", "#CA6705", "#EE9B04", "#EAD7A4", "#CAB67B", "#A99945", "#897C0F", "#736E12", "#5E6014"],
        "OrBu": ["#B97401", "#DC8D01", "#FAC316", "#F8E584", "#F8FFC9", "#638094", "#4C6E83", "#3E596D", "#374053"],
        "OrGn": ["#B97401", "#DC8D01", "#FAC316", "#F8E584", "#F1FFC1", "#7DA4A4", "#628E8E", "#467171", "#284C4C"],
        "PiBu": ["#7D433D", "#A45040", "#C45D47", "#DD6850", "#CAA59F", "#4C8A9A", "#427283", "#385B6C", "#2E4355"],
        "gruvbox_light": ["#FB4934", "#FE8019", "#FABD2F", "#B8BB26", "#8EC07C", "#83A598", "#D3869B"],
        "gruvbox": ["#CC241D", "#D65D0E", "#D79921", "#98971A", "#689D6A", "#458588", "#B16286"],
        "gruvbox_dark": ["#9D0006", "#AF3A03", "#B57614", "#79740E", "#427B58", "#076678", "#8F3F71"],
        "artdeco": ["#9F1B10", "#D88533", "#E8B055", "#D9B97B", "#B6BEAA", "#768C86", "#365861", "#204755", "#0A3649"],
        "cyberpunk": ["#55D6F5", "#5C9BE8", "#6260DC", "#522FAA", "#42007A", "#4F057A", "#5D097C", "#A917BE", "#F225FF"],
        "synthwave": ["#5C9BE8", "#5368C4", "#4B35A0", "#42007A", "#550584", "#680B8E", "#7B1098", "#A75466", "#D19536", "#FBD606"],
        "aquamarine_light": ["#7E3FE8", "#6B5CE6", "#577CE2", "#439BDF", "#2DBDDC", "#17DCD4", "#00E8BE"],
        "aquamarine": ["#7239D2", "#5F52CC", "#4C6CC5", "#3985BF", "#269FB9", "#13B9B2", "#00D2AC"],
        "aquamarine_dark": ["#391C69", "#2F2966", "#263662", "#1C4360", "#13505C", "#095C59", "#006956"],
        # "aquamarine_light": aquamarine_light,
        # "aquamarine": aquamarine,
        # "aquamarine_dark": aquamarine_dark,
    }
    for name, colors in color_dict.items():
        try:
            colors_ = colors[::-1]
            name_ = name.split("_")[0] + "_r"
            if name.find("_") > 0:
                name_ = name_ + "_" + "_".join(name.split("_")[1:])
            cmap = ListedColormap(colors_, name_)
            _register_cmap(cmap)
        except:
            pass
    for name, colors in color_dict.items():
        try:
            cmap = ListedColormap(colors, name)
            _register_cmap(cmap)
        except:
            pass
    return


def _get_bgcolor_dict():
    """
    Gets bockground color dictionary

    """
    bg_color_dict = {
        "gruvbox": "#1D2021",
        "gruvbox_dark": "#1D2021",
        "gruvbox_light": "#FBF1C7",
    }
    return bg_color_dict


def _get_darkest(cmap):
    """
    Gets darkest color from colormap and darkens it even more

    Parameters
    ----------
    cmap : str or matplotlib.colors.Colormap object
        Colormap

    Returns
    -------
    color : str
        Color in hex format

    """
    cmap = plt.get_cmap(cmap)
    colors = get_colors(cmap)
    h, s, v = np.array([mc.rgb_to_hsv(mc.to_rgb(c)) for c in colors]).T
    color = colors[np.argmin(v)]
    return color










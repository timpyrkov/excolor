#!/usr/bin/env python
# -*- coding: utf8 -*-

import numpy as np
import pylab as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.colors as mc
from cycler import cycler
from excolor.utils import _is_arraylike, _is_int, _is_exp, _is_log
from excolor.utils import _is_qualitative, _is_cyclic, _is_divergent
from excolor.background import *

import warnings
warnings.filterwarnings("ignore")


def get_colors(cmap, n=None, exclude_extreme=True, mode="hex"):
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
    mode : {"hex", "rgb", "rgba", "hsv"}, default "hex"
        Format of colors

    Returns
    -------
    colors : list
        List of colors

    """
    assert mode in ["hex", "rgb", "rgba", "hsv"]
    cmap = plt.get_cmap(cmap)
    if _is_qualitative(cmap):
        colors = cmap.colors
    else:
        n = 10 - _is_divergent(cmap) if n is None else n
        dn = 1 if exclude_extreme else 0
        idx = np.arange(dn, n + dn) / (n + 2 * dn - 1)
        colors = cmap(idx)
    colors = [mc.to_hex(c, keep_alpha=False).upper() for c in colors]
    if mode == "rgb":
        colors = [mc.to_rgb(c) for c in colors]
    elif mode == "rgba":
        colors = [mc.to_rgba(c) for c in colors]
    elif mode == "hsv":
        colors = [mc.rgb_to_hsv(mc.to_rgb(c)) for c in colors]
    return colors


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
        gradient = np.arange(10)
    cname = [f"{g + 1:.0f}" for g in gradient]
    if _is_exp(gradient):
        cname = [f"{g:.2e}" for g in gradient]
    gradient = gradient / np.max(gradient)
    colors = plt.get_cmap(cmap)(gradient)
    show_colors(colors, cmap.name, cname)
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
    plt.show()
    return


def show_colors(c, title="", cname=None):
    """
    Plots colors from list or a colormap

    Parameters
    ----------
    c : list, str, or matplotlib.colors.Colormap object
        List of colors or a colormap
    title : str, optional
        Figure title
    cname : list, optional
        List of color names

    """
    try:
        c = plt.get_cmap(c)
        colors = get_colors(c, exclude_extreme=False)
        title = c.name
        is_cmap = True
    except:
        colors = c if _is_arraylike(c) else [c]
        is_cmap = False
    d = 0.05
    width = 1  -2 * d
    if cname is None:
        cname = [color_to_hex(color) for color in colors]
    n, m = aspect_ratio(len(colors))
    plt.figure(figsize=(2*n+4,2*m), facecolor="#00000000")
    plt.title(title, fontsize=20, color="grey")
    for k, color in enumerate(colors):
        i = k % n
        j = k // n
        r = Rectangle((i, -j), width, -width, facecolor=color, fill=True)
        plt.gca().add_patch(r)
        h, s, v = mc.rgb_to_hsv(mc.to_rgb(color)).T
        fontcolor = "white" if v < 0.6 else "black"
        x, y = i + 0.5 - 2 * d, -j - 0.5
        plt.text(x, y, cname[k], fontsize=20, color=fontcolor, ha="center")
    plt.xlim(-d, n - d)
    plt.ylim(-m + d, d)
    plt.gca().set_axis_off()
    plt.tight_layout()
    plt.show()
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


def lighten(c, scale=0.5):
    """
    Lightens colors or colormap

    Parameters
    ----------
    c : list, str, or matplotlib.colors.Colormap object
        List of colors or a colormap
    scale : float, optional
        Scale in range (0,1) - propensity of lightening

    Returns
    -------
    cmod : list or matplotlib.colors.Colormap object
        List of modified colors or colormap

    """
    try:
        c = plt.get_cmap(c)
        colors = get_colors(c, 256, exclude_extreme=False)
        is_cmap = True
    except:
        colors = c if _is_arraylike(c) else [c]
        is_cmap = False
    hsv = np.array([mc.rgb_to_hsv(mc.to_rgb(color)) for color in colors]).T
    hsv[2] = hsv[2] + scale * (1 - hsv[2])
    cmod = [mc.hsv_to_rgb(color) for color in hsv.T]
    if is_cmap:
        name = c.name + "_light"
        cmod = LinearSegmentedColormap.from_list(name, cmod)
    elif len(cmod) < 2:
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
        is_cmap = True
    except:
        colors = c if _is_arraylike(c) else [c]
        is_cmap = False
    hsv = np.array([mc.rgb_to_hsv(mc.to_rgb(color)) for color in colors]).T
    hsv[2] = (1 - scale) * hsv[2]
    cmod = [mc.hsv_to_rgb(color) for color in hsv.T]
    if is_cmap:
        name = c.name + "_dark"
        cmod = LinearSegmentedColormap.from_list(name, cmod)
    elif len(cmod) < 2:
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
        is_cmap = True
    except:
        colors = c if _is_arraylike(c) else [c]
        is_cmap = False
    hsv = np.array([mc.rgb_to_hsv(mc.to_rgb(color)) for color in colors]).T
    hsv[1] = hsv[1] + scale * (1 - hsv[1])
    cmod = [mc.hsv_to_rgb(color) for color in hsv.T]
    if is_cmap:
        name = c.name + "_saturated"
        cmod = LinearSegmentedColormap.from_list(name, cmod)
    elif len(cmod) < 2:
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
        is_cmap = True
    except:
        colors = c if _is_arraylike(c) else [c]
        is_cmap = False
    hsv = np.array([mc.rgb_to_hsv(mc.to_rgb(color)) for color in colors]).T
    hsv[1] = (1 - scale) * hsv[1]
    cmod = [mc.hsv_to_rgb(color) for color in hsv.T]
    if is_cmap:
        name = c.name + "_desaturated"
        cmod = LinearSegmentedColormap.from_list(name, cmod)
    elif len(cmod) < 2:
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
    hexname = mc.to_hex(color, keep_alpha).upper()
    return hexname


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


def add_cool_warm_colormaps():
    """
    Extends list of registered colormaps by modifications of 'coolwarm'

    """
    # Divergent to sequential
    gradient = np.linspace(0.5, 1, 128)
    warm_colors = plt.get_cmap("coolwarm")(gradient)
    cool_colors = plt.get_cmap("coolwarm_r")(gradient)
    cmap = LinearSegmentedColormap.from_list("warm", warm_colors)
    plt.colormaps.register(cmap)
    cmap = LinearSegmentedColormap.from_list("warm_r", warm_colors[::-1])
    plt.colormaps.register(cmap)
    cmap = LinearSegmentedColormap.from_list("cold", cool_colors)
    plt.colormaps.register(cmap)
    cmap = LinearSegmentedColormap.from_list("cold_r", cool_colors[::-1])
    plt.colormaps.register(cmap)
    # Log-scaled
    gradient = np.logspace(0, -10, 128)
    logwarm_colors = plt.get_cmap("warm")(gradient)
    logwarm_r_colors = plt.get_cmap("warm_r")(gradient)
    cmap = LinearSegmentedColormap.from_list("logwarm", logwarm_colors)
    plt.colormaps.register(cmap)
    cmap = LinearSegmentedColormap.from_list("logwarm_r", logwarm_r_colors)
    plt.colormaps.register(cmap)
    logcool_colors = plt.get_cmap("cold")(gradient)
    logcool_r_colors = plt.get_cmap("cold_r")(gradient)
    cmap = LinearSegmentedColormap.from_list("logcool", logcool_colors)
    plt.colormaps.register(cmap)
    cmap = LinearSegmentedColormap.from_list("logcool_r", logcool_r_colors)
    plt.colormaps.register(cmap)
    return


def add_extended_colormaps():
    """
    Extends list of registered colormaps by modifications of 'coolwarm'
    and manually hardcoded colormaps

    """
    try:
        add_cool_warm_colormaps()
    except:
        pass
    aquamarine = grey_to_aquamarine()
    aquamarine_light = lighten(aquamarine, 0.8)
    aquamarine_dark = darken(aquamarine, 0.4)
    color_dict = {
        "gruvbox_light": ["#FB4934", "#FE8019", "#FABD2F", "#B8BB26", "#8EC07C", "#83A598", "#D3869B"],
        "gruvbox": ["#CC241D", "#D65D0E", "#D79921", "#98971A", "#689D6A", "#458588", "#B16286"],
        "gruvbox_dark": ["#9D0006", "#AF3A03", "#B57614", "#79740E", "#427B58", "#076678", "#8F3F71"],
        "synthwave": ["#A148AB", "#DD517F", "#E68E36", "#EAC180", "#7998EE", "#556DC8"],
        "aquamarine_light": ["#7E3FE8", "#419AE0", "#00E8BE"],
        "aquamarine": ["#7239D2", "#3884C0", "#00D2AC"],
        "aquamarine_dark": ["#391C69", "#1C4260", "#006956"],
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
            plt.colormaps.register(cmap)
        except:
            pass
    for name, colors in color_dict.items():
        try:
            cmap = ListedColormap(colors, name)
            plt.colormaps.register(cmap)
        except:
            pass
    return


def aspect_ratio(length):
    """
    Finds optimal aspect ratio to represent sequence of charts

    Parameters
    ----------
    length : int
        Number of items

    Returns
    -------
    n : int
       Number of columns
    m : int
        Number of rows

    """
    d = np.array([-2, -1, 0, 1, 2])
    n0 = np.sqrt(length / 2)
    ns = []
    ms = []
    ds = []
    for s in [4, 5, 6]:
        n1 = (2 * n0 // s + d).astype(int) * s
        m1 = np.ceil(length / n1).astype(int)
        for k in range(5):
            if n1[k] > 0 and m1[k] > 0 and n1[k] > m1[k]:
                ns.append(n1[k])
                ms.append(m1[k])
                ds.append(np.abs(length - n1[k] * m1[k]))
    mask = np.array(ds) == min(ds)
    ns = np.array(ns)[mask]
    ms = np.array(ms)[mask]
    idx = np.argmin(np.abs(ns / ms - 6.5))
    n, m = ns[idx], ms[idx]
    if isinstance(n, np.ndarray) and len(n) > 1:
        n, m = n[0], m[0]
    return n, m


""" Aliases for functions """
show_colormap = show_cmap
show_colorbar = show_cbar
gray_to_hue = grey_to_hue
gray_to_aquamarine = grey_to_aquamarine

import types
__all__ = [name for name, thing in globals().items()
          if not (name.startswith("_") or isinstance(thing, types.ModuleType))]
del types


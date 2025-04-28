#!/usr/bin/env python
# -*- coding: utf8 -*-

"""
This module contains utility functions for colortools and cmaptools.
"""

import numpy as np
from typing import Union, Optional, List
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from matplotlib.colors import LinearSegmentedColormap, Colormap

import warnings
warnings.filterwarnings("ignore")



def _is_qualitative(cmap: Union[str, Colormap]) -> bool:
    """
    Tests if a colormap is qualitative (categorical).

    This function determines whether a colormap is qualitative, meaning it is
    designed for categorical data with distinct colors rather than continuous
    gradients. Qualitative colormaps typically have a limited number of colors
    (usually 20 or fewer) and are not meant to be interpolated.

    Parameters
    ----------
    cmap : str or matplotlib.colors.Colormap
        Colormap to check for qualitative properties

    Returns
    -------
    bool
        True if the colormap is qualitative,
        False otherwise

    Examples
    --------
    >>> _is_qualitative('Set1')  # True
    >>> _is_qualitative('viridis')  # False
    >>> _is_qualitative(plt.cm.Set1)  # True
    """
    cmap = plt.get_cmap(cmap)
    try:
        colors = cmap.colors
        mask = len(colors) <= 20
    except:
        mask = False
    return mask


def _is_cyclic(cmap: Union[str, Colormap]) -> bool:
    """
    Tests if a colormap is cyclic.

    A cyclic colormap is one where the colors at both ends are similar,
    making it suitable for visualizing periodic data. This function checks
    if the Hue, Saturation, and Value components are approximately the same
    at both ends of the colormap.

    Parameters
    ----------
    cmap : str or matplotlib.colors.Colormap
        Colormap to check for cyclic properties

    Returns
    -------
    bool
        True if the colormap is cyclic,
        False otherwise

    Examples
    --------
    >>> _is_cyclic('twilight')  # True
    >>> _is_cyclic('viridis')  # False
    >>> _is_cyclic(plt.cm.twilight)  # True
    """
    n = 32
    colors = get_colors(cmap, n, exclude_extreme=False)
    hsv = np.array([mc.rgb_to_hsv(mc.to_rgb(color)) for color in colors])
    d = np.abs(hsv[-1] - hsv[0])
    d[0] = min(d[0], abs(d[0] - 1), abs(d[0] + 1))
    mask = np.all(d < 0.1) & (~_is_qualitative(cmap))
    return mask


def _is_divergent(cmap: Union[str, Colormap]) -> bool:
    """
    Tests if a colormap is divergent.

    A divergent colormap is one that has a distinct middle color (often white or gray)
    and diverges to two different colors at the extremes. This function checks for
    divergent properties by analyzing the hue, saturation, and value components
    of the colormap.

    Parameters
    ----------
    cmap : str or matplotlib.colors.Colormap
        Colormap to check for divergent properties

    Returns
    -------
    bool
        True if the colormap is divergent,
        False otherwise

    Examples
    --------
    >>> _is_divergent('RdBu')  # True
    >>> _is_divergent('viridis')  # False
    >>> _is_divergent(plt.cm.RdBu)  # True
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


def get_colors(
    cmap: Union[str, Colormap],
    n: Optional[int] = None,
    exclude_extreme: bool = True
) -> List[str]:
    """
    Extracts colors from a colormap with optional sampling and filtering.

    This function extracts colors from a colormap, with options to:
    - Specify the number of colors to extract
    - Exclude extreme (very dark/light) colors
    - Handle both qualitative and continuous colormaps appropriately

    Parameters
    ----------
    cmap : str or matplotlib.colors.Colormap
        Colormap name or instance
    n : int, optional
        Number of colors to extract. If None:
        - For qualitative colormaps: uses all colors
        - For continuous colormaps: uses 10 colors (9 for divergent)
    exclude_extreme : bool, default=True
        Whether to exclude the darkest and lightest colors from the output.
        This is useful for continuous colormaps to avoid pure black/white.

    Returns
    -------
    list of str
        List of colors in hex format, sampled from the colormap

    Examples
    --------
    >>> get_colors('viridis', n=5)  # Get 5 colors from viridis
    >>> get_colors('Set1')  # Get all colors from qualitative colormap
    >>> get_colors('RdBu', exclude_extreme=False)  # Include extremes
    """
    cmap = plt.get_cmap(cmap)
    if _is_qualitative(cmap):
        colors = cmap.colors
        if n is not None:
            colors = interpolate_colors(colors, n=10*len(colors))
            cmap = LinearSegmentedColormap.from_list("cmap", colors)
            colors = get_colors(cmap, n, exclude_extreme=False)
    else:
        n = 10 - _is_divergent(cmap) if n is None else n
        dn = 1 if exclude_extreme else 0
        idx = np.arange(dn, n + dn) / (n + 2 * dn - 1)
        colors = cmap(idx)
    colors = [mc.to_hex(c, keep_alpha=False).upper() for c in colors]
    return colors


def interpolate_colors(c: List[str], n: int = 5) -> List[str]:
    """
    Creates a smooth gradient of colors by interpolating between input colors.

    This function takes a list of colors and generates a new list with n colors
    by creating a smooth gradient between the input colors using linear interpolation.

    Parameters
    ----------
    c : list of str
        List of input colors in any matplotlib-compatible format (hex, name, etc.)
    n : int, default=5
        Number of colors to generate in the output list

    Returns
    -------
    list of str
        List of n colors in hex format, forming a smooth gradient between the
        input colors

    Examples
    --------
    >>> interpolate_colors(['#FF0000', '#00FF00'], n=3)
    ['#FF0000', '#808000', '#00FF00']
    >>> interpolate_colors(['red', 'blue'], n=4)
    ['#FF0000', '#800080', '#0000FF']
    """
    gradient = np.linspace(0,1,n)
    cmap = LinearSegmentedColormap.from_list("cmap", c)
    colors = cmap(gradient)
    colors = [mc.to_hex(color).upper() for color in colors]
    return colors


import types
__all__ = [name for name, thing in globals().items()
          if not (name.startswith("_") or isinstance(thing, types.ModuleType))]
del types

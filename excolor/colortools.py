#!/usr/bin/env python
# -*- coding: utf8 -*-

"""
This module contains functions to manipulate colors.
"""

import numpy as np
import pylab as plt
import colorsys
from cycler import cycler
import matplotlib.colors as mc
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, Colormap
from matplotlib.patches import Rectangle
from .utils import _is_qualitative, get_colors
from typing import Union, Optional, Tuple, List, Any

import warnings
warnings.filterwarnings("ignore")


def _is_cmap(c: Union[str, Colormap]) -> bool:
    """
    Tests if the argument is a valid colormap name or matplotlib.colors.Colormap object.

    This function checks whether the input can be used as a colormap in matplotlib.
    It accepts both colormap names (strings) and Colormap objects.

    Parameters
    ----------
    c : str or matplotlib.colors.Colormap
        Input to check for colormap validity

    Returns
    -------
    bool
        True if c is a valid colormap name or Colormap object,
        False otherwise

    Examples
    --------
    >>> _is_cmap('viridis')  # True
    >>> _is_cmap(plt.cm.viridis)  # True
    >>> _is_cmap('not_a_colormap')  # False
    >>> _is_cmap(42)  # False
    """
    try:
        is_cmap = False
        if c is not None and c not in mc.CSS4_COLORS:
            c = plt.get_cmap(c)
            is_cmap = True
    except:
        is_cmap = False
    return is_cmap


def _is_arraylike(x: Any) -> bool:
    """
    Checks if an object is array-like (can be treated as a sequence of elements).

    This function tests whether an object can be treated as an array or sequence,
    supporting operations like indexing and iteration. It checks for common
    array-like types in Python and NumPy.

    Parameters
    ----------
    x : Any
        Object to check for array-like properties

    Returns
    -------
    bool
        True if x is array-like (numpy.ndarray, list, tuple, or set),
        False otherwise

    Examples
    --------
    >>> _is_arraylike([1, 2, 3])  # True
    >>> _is_arraylike(np.array([1, 2, 3]))  # True
    >>> _is_arraylike((1, 2, 3))  # True
    >>> _is_arraylike({1, 2, 3})  # True
    >>> _is_arraylike(42)  # False
    """
    mask = isinstance(x, np.ndarray) or isinstance(x, list)
    mask = mask or isinstance(x, tuple) or isinstance(x, set)
    return mask


def _is_rgb(x: Any) -> bool:
    """
    Checks if an object is an RGB or RGBA like array.

    This function tests whether an object can be treated as an RGB or RGBA array.

    Parameters
    ----------
    x : Any
        Object to check for RGB or RGBA properties

    Returns
    -------
    bool
        True if c is an RGB or RGBA array, False otherwise

    Examples
    --------
    >>> _is_rgb((1.0, 0.0, 0.0))  # True
    >>> _is_rgb((1.0, 0.0, 0.0, 1.0))  # True
    >>> _is_rgb('red')  # False
    """ 
    mask = _is_arraylike(x) and len(x) in [3, 4]
    mask = mask and all([isinstance(x_, float) or isinstance(x_, int) for x_ in x])
    mask = mask and all([0 <= x_ <= 255 for x_ in x])
    return mask


def _aspect_ratio(length: int, lmin: int = 0) -> Tuple[int, int]:
    """
    Calculates the optimal grid dimensions for displaying a sequence of items.

    This function determines the best number of rows and columns to arrange
    a given number of items in a grid layout, with the goal of creating a
    visually pleasing aspect ratio close to 5:1 (width:height).

    Parameters
    ----------
    length : int
        Total number of items to arrange in the grid
    lmin : int, default=0
        Minimum number of items required to split into multiple rows.
        If length <= lmin, all items will be placed in a single row.

    Returns
    -------
    tuple of int
        A tuple (n, m) where:
        - n: Number of columns in the grid
        - m: Number of rows in the grid

    Notes
    -----
    The function:
    1. Calculates an initial estimate for the number of columns
    2. Tests various grid configurations around this estimate
    3. Selects the configuration that:
       - Minimizes empty spaces
       - Has an aspect ratio closest to 5:1
       - Has more columns than rows

    Examples
    --------
    >>> _aspect_ratio(12)  # For 12 items
    (6, 2)  # 6 columns, 2 rows
    >>> _aspect_ratio(5, lmin=6)  # Few items, below minimum
    (5, 1)  # Single row
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


def show_colors(
    c: Union[Colormap, str, List[str], Tuple[float, ...], List[Tuple[float, ...]]] = None,
    names: Optional[List[str]] = None,
    title: str = "",
    size: Optional[Tuple[int, int]] = None,
    fmt: str = 'hex',
    verbose: bool = True
) -> None:
    """
    Displays a set of colors in a grid layout with their hex values.

    This function creates a visualization of colors, either from a list, a single color,
    or a colormap. The colors are displayed in a grid with their hex values, and the
    text color is automatically chosen for readability based on the background color.

    Parameters
    ----------
    c : list of str or tuple, str or tuple, or matplotlib.colors.Colormap, optional
        Input colors to display. Can be:
        - A colormap name or instance
        - A single color str or rgb tuple
        - A list of colors str or rgb tuples
        If None, the matplotlib default color palettes will be used.
    names : list of str, optional
        List of color names. If not provided, the hex values will be used.
    title : str, default=''
        Title to display above the color grid
    size : tuple of int, optional
        Size of the color grid (width, height)
    fmt: str, default='hex'
        Output format of color names ('hex', 'rgb', 'hsv', 'hsl')
    verbose : bool, default=True
        If True, prints the list of colors to the console

    Returns
    -------
    None
        Displays the color visualization using matplotlib

    Examples
    --------
    >>> show_colors(['#FF0000', '#00FF00', '#0000FF'])  # Display RGB colors
    >>> show_colors('viridis')  # Display viridis colormap colors
    >>> show_colors('viridis', size=(10, 5))  # Custom size
    >>> show_colors(None)  # Display matplotlib default color palettes
    """
    if c is None:
        list_colors()
        return
    if _is_arraylike(c) and len(c) == 0:
        list_colors()
        return
    def _to_255(x):
        x255 = tuple([int(np.round(x_ * 255)) for x_ in x])
        return x255
    def _to_name(x: Tuple[float, ...]) -> str:
        name = [f'{int(np.round(x_ * 255)):3d}' for x_ in x]
        name = '  ' + ' '.join(name)
        return name
    try:
        c = plt.get_cmap(c)
        colors = get_colors(c, exclude_extreme=False)
        title = c.name
    except:
        colors = c if _is_arraylike(c) and not _is_rgb(c) else [c]
    rgbcolors = [to_rgb(color) for color in colors]
    if fmt == 'hex':
        colors = [to_hex(color) for color in colors]
        cnames = colors
    elif fmt == 'rgb':
        colors = rgbcolors
        cnames = [_to_name(color) for color in colors]
    elif fmt == 'hsv':
        colors = [colorsys.rgb_to_hsv(*to_rgb(color)) for color in colors]
        cnames = [_to_name(color) for color in colors]
    elif fmt == 'hsl':
        colors = [colorsys.rgb_to_hls(*to_rgb(color)) for color in colors]
        cnames = [_to_name(color) for color in colors]
    if verbose:
        if fmt == 'hex':
            print(colors)
        else:
            print([_to_255(color) for color in colors])
    d = 0.05
    width = 1 - 2 * d
    if size is None:
        n, m = _aspect_ratio(len(rgbcolors), lmin=12)
        size = (2*n+4,2*m)
    else:
        n, m = size
        m = np.ceil(len(rgbcolors) / n)
        size = (2*n+4,2*m)
    fontsize = 12 if size[1] < 2 else 28
    plt.figure(figsize=size, facecolor="#00000000")
    plt.title(title, fontsize=fontsize, color="grey")
    for k, rgb in enumerate(rgbcolors):
        i = k % n
        j = k // n
        if rgb is not None:
            r = Rectangle((i, -j), width, -width, facecolor=rgb, fill=True)
            plt.gca().add_patch(r)
            h, s, v = to_hsv(rgb)
            h, l, s = to_hls(rgb)
            fontcolor = "white" if v < 0.4 or l < 0.3 else "black"
            x, y = i + 0.55 - 2 * d, -j - 0.5
            name = names[k] if names is not None else cnames[k]
            plt.text(x, y, name, fontsize=20, color=fontcolor, ha="center", va="center")
    plt.xlim(-d, n - d)
    plt.ylim(-m + d, d)
    plt.gca().set_axis_off()
    plt.tight_layout()
    plt.show()
    plt.close()
    return


def list_colors():
    """
    Displays a list of colors in a grid layout with their hex values.

    This function creates a visualization of colors, either from a dictionary,
    or from the default color palettes. The colors are displayed in a grid with
    their hex values, and the text color is automatically chosen for readability
    based on the background color.

    Returns
    -------
    None
        Displays the color visualization using matplotlib
    """
    groups = {
        'Reds': [0.000, 0.050],
        'Oranges': [0.050, 0.110],
        'Yellows': [0.110, 0.150],
        'Yellow-Greens': [0.150, 0.220],
        'Greens': [0.220, 0.440],
        'Cyans': [0.440, 0.530],
        'Blues': [0.530, 0.730],
        'Violets': [0.730, 0.880],
        'Purples': [0.800, 1.100],
    }
    dcts = {
        'Base Colors': mc.BASE_COLORS,
        'Tableau Palette': mc.TABLEAU_COLORS,
        'CSS Colors': mc.CSS4_COLORS,
        'XKCD Colors': mc.XKCD_COLORS,
    }
    size = (10,1) # width, height
    for title, dct in dcts.items():
        
        cname = np.array(list(dct.keys()))
        color = np.array(list(dct.values()))
        hue, lightness, saturation = np.array([to_hls(v) for v in color]).T
    
        # Show small palettes 
        if len(color) < 20:
            subtitle = f'{title}'
            value = hue + (saturation > 0).astype(float)
            idx = np.argsort(value)
            cnames = cname[idx]
            colors = color[idx]
            hexnames = [to_hex(c) for c in colors]
            cnames = [f'{cnames[i]}\n{hexnames[i]}' for i in range(len(colors))]
            show_colors(colors, names=cnames, size=size, title=subtitle)

        # Show large palettes by hue groups
        else:
            # Show grays
            subtitle = f'{title} (Grays)'
            mask = saturation == 0
            cnames = cname[mask]
            colors = color[mask]
            lights = lightness[mask]
            idx = np.argsort(lights)
            cnames = cnames[idx]
            colors = colors[idx]
            if title.startswith('CSS'):
                cnames = [f'{cnames[i]}\n{colors[i]}' for i in range(len(colors))]
            else:
                cnames = [f'xkcd:\n{cnames[i][5:]}\n{colors[i]}' for i in range(len(colors))]
            show_colors(colors, names=cnames, size=size, title=subtitle)

            # Show color groups by hue
            for group, bins in groups.items():
                subtitle = f'{title} ({group})'
                hue0, hue1 = bins
                mask = (hue >= hue0) & (hue < hue1) & (saturation > 0)
                cnames = cname[mask]
                colors = color[mask]
                lights = lightness[mask]
                idx = np.argsort(lights)
                cnames = cnames[idx]
                colors = colors[idx]
                if title.startswith('CSS'):
                    cnames = [f'{cnames[i]}\n{colors[i]}' for i in range(len(colors))]
                else:
                    cnames = [f'xkcd:\n{cnames[i][5:]}\n{colors[i]}' for i in range(len(colors))]
                show_colors(colors, names=cnames, size=size, title=subtitle)

    return


def set_color_cycler(c: Union[List[str], str, Colormap], n: int = 3) -> None:
    """
    Sets the color cycler for the current matplotlib axis.

    This function configures the color cycling behavior for the current matplotlib
    axis using either a list of colors or a colormap. The colors will be used in
    sequence when plotting multiple lines or other elements.

    Parameters
    ----------
    c : list of str, str, or matplotlib.colors.Colormap
        Input colors to use in the cycler. Can be:
        - A list of color strings or rgb tuples
        - A colormap name or instance
    n : int, default=3
        Number of colors to extract from the colormap if a colormap is provided.
        If a list of colors is provided, this parameter is ignored.

    Returns
    -------
    None
        Modifies the current matplotlib axis's color cycler

    Examples
    --------
    >>> set_color_cycler(['red', 'green', 'blue'])  # Use specific colors
    >>> set_color_cycler('viridis', n=5)  # Use 5 colors from viridis
    >>> plt.plot(x1, y1)  # Will use first color
    >>> plt.plot(x2, y2)  # Will use second color
    """
    try:
        c = plt.get_cmap(c)
        colors = get_colors(c, n, exclude_extreme=True)
    except:
        colors = c if _is_arraylike(c) and not _is_rgb(c) else None
    plt.gca().set_prop_cycle(cycler(color=colors))
    return


def lighten(
    c: Union[Colormap, str, List[str], Tuple[float, ...], List[Tuple[float, ...]]],
    factor: float = 0.1,
    keep_alpha: bool = False,
    mode: str = 'hls'
) -> Union[str, List[str], Tuple[float, ...], List[Tuple[float, ...]]]:
    """
    Lightens colors or a colormap by adjusting their lightness.

    This function takes colors or a colormap and returns lighter versions by
    increasing their lightness in HLS color space. 

    Parameters
    ----------
    c : list of str or tuple, str or tuple, or matplotlib.colors.Colormap
        Input colors to display. Can be:
        - A colormap name or instance
        - A single color str or rgb tuple
        - A list of colors str or rgb tuples
    factor : float, default=0.2
        Increment in lightness between 0 and 1:
    keep_alpha : bool, default=False
        If True, keep the alpha channel
    mode : str, default='hls'
        If 'hls' or 'hsl', use HLS color space to lighten the colors
        If 'hsv', use HSV color space to lighten the colors

    Returns
    -------
    list of str or tuple, str or tuple, or matplotlib.colors.Colormap
        Lightened version of the input color or colors.
        Returns same type as input.

    Examples
    --------
    >>> lighten('#FF0000', factor=0.5)  # Lighten red
    >>> lighten(['#FF0000', '#00FF00'], factor=0.3)  # Lighten multiple colors
    >>> lighten('viridis', factor=0.3)  # Lighten entire colormap
    """
    if mode not in ['hls', 'hsl', 'hsv']:
        raise ValueError("mode must be 'hls' or 'hsv'")
    factor = np.clip(factor, 0, 1)
    try:
        if _is_cmap(c):
            c = plt.get_cmap(c)
            colors = get_colors(c, exclude_extreme=False)
            if not _is_qualitative(c):
                colors = get_colors(c, 256, exclude_extreme=False)
            category = 'cmap'
            print(category)
        elif _is_arraylike(c) and not _is_rgb(c):
            colors = c
            category = 'hex' if isinstance(c[0], str) else 'rgb'
        else:
            colors = [c]
            category = 'hex' if isinstance(c, str) else 'rgb'
    except:
        colors = None
        category = None
    if colors is not None and len(colors) > 1:
        colors = [lighten(color, factor, keep_alpha, mode) for color in colors]
        if category == 'cmap':
            name = c.name + "_light"
            if _is_qualitative(c):
                colors = ListedColormap(colors, name=name)
            else:
                colors = LinearSegmentedColormap.from_list(name, colors)
    else:
        try:
            if mode == 'hsv':
                hsv = np.array(to_hsv(colors[0], keep_alpha=keep_alpha))
                hsv[2] = np.clip(hsv[2] + factor, 0, 1)
                rgb = mc.hsv_to_rgb(hsv[:3])
                rgb = rgb + (hsv[3],) if hsv.shape[0] == 4 else rgb
                rgb = tuple([float(np.round(x, 3)) for x in rgb])
            elif mode in ['hls', 'hsl']:
                hls = np.array(to_hls(colors[0], keep_alpha=keep_alpha))
                hls[1] = np.clip(hls[1] + factor, 0, 1)
                rgb = colorsys.hls_to_rgb(*hls[:3])
                rgb = rgb + (hls[3],) if hls.shape[0] == 4 else rgb
                rgb = tuple([float(np.round(x, 3)) for x in rgb])
            if category == 'hex':
                colors = to_hex(rgb, keep_alpha=keep_alpha).upper()
            else:
                colors = to_rgb(rgb, keep_alpha=keep_alpha)
        except:
            colors = None
    return colors


def darken(
    c: Union[Colormap, str, List[str], Tuple[float, ...], List[Tuple[float, ...]]],
    factor: float = 0.1,
    keep_alpha: bool = False,
    mode: str = 'hls'
) -> Union[str, List[str], Tuple[float, ...], List[Tuple[float, ...]]]:
    """
    Darkens colors or a colormap by adjusting their lightness.

    This function takes colors or a colormap and returns darker versions by
    decreasing their lightness in HLS color space. 

    Parameters
    ----------
    c : list of str or tuple, str or tuple, or matplotlib.colors.Colormap
        Input colors to display. Can be:
        - A colormap name or instance
        - A single color str or rgb tuple
        - A list of colors str or rgb tuples
    factor : float, default=0.2
        Decrement in lightness between 0 and 1:
    keep_alpha : bool, default=False
        If True, keep the alpha channel
    mode : str, default='hls'
        If 'hls' or 'hsl', use HLS color space to darken the colors
        If 'hsv', use HSV color space to darken the colors

    Returns
    -------
    list of str or tuple, str or tuple, or matplotlib.colors.Colormap
        Darkened version of the input color or colors.
        Returns same type as input.

    Examples
    --------
    >>> darken('#FF0000', factor=0.5)  # Darken red
    >>> darken(['#FF0000', '#00FF00'], factor=0.3)  # Darken multiple colors
    >>> darken('viridis', factor=0.3)  # Darken entire colormap
    """
    if mode not in ['hls', 'hsl', 'hsv']:
        raise ValueError("mode must be 'hls' or 'hsv'")
    factor = np.clip(factor, 0, 1)
    try:
        if _is_cmap(c):
            c = plt.get_cmap(c)
            colors = get_colors(c, exclude_extreme=False)
            if not _is_qualitative(c):
                colors = get_colors(c, 256, exclude_extreme=False)
            category = 'cmap'
        elif _is_arraylike(c) and not _is_rgb(c):
            colors = c
            category = 'hex' if isinstance(c[0], str) else 'rgb'
        else:
            colors = [c]
            category = 'hex' if isinstance(c, str) else 'rgb'
    except:
        colors = None
        category = None
    if colors is not None and len(colors) > 1:
        colors = [darken(color, factor, keep_alpha, mode) for color in colors]
        if category == 'cmap':
            name = c.name + "_dark"
            if _is_qualitative(c):
                colors = ListedColormap(colors, name=name)
            else:
                colors = LinearSegmentedColormap.from_list(name, colors)
    else:
        try:
            if mode == 'hsv':
                hsv = np.array(to_hsv(colors[0], keep_alpha=keep_alpha))
                hsv[2] = np.clip(hsv[2] - factor, 0, 1)
                rgb = mc.hsv_to_rgb(hsv[:3])
                rgb = rgb + (hsv[3],) if hsv.shape[0] == 4 else rgb
                rgb = tuple([float(np.round(x, 3)) for x in rgb])
            elif mode in ['hls', 'hsl']:
                hls = np.array(to_hls(colors[0], keep_alpha=keep_alpha))
                hls[1] = np.clip(hls[1] - factor, 0, 1)
                rgb = colorsys.hls_to_rgb(*hls[:3])
                rgb = rgb + (hls[3],) if hls.shape[0] == 4 else rgb
            if category == 'hex':
                colors = to_hex(rgb, keep_alpha=keep_alpha).upper()
            else:
                colors = to_rgb(rgb, keep_alpha=keep_alpha)
        except:
            colors = None
    return colors


def saturate(
    c: Union[Colormap, str, List[str], Tuple[float, ...], List[Tuple[float, ...]]],
    factor: float = 0.1,
    keep_alpha: bool = False,
    mode: str = 'hls'
) -> Union[str, List[str], Tuple[float, ...], List[Tuple[float, ...]]]:
    """
    Saturates colors or a colormap by adjusting their saturation.

    This function takes colors or a colormap and returns more saturated versions by
    increasing their saturation in HLS color space. 

    Parameters
    ----------
    c : list of str or tuple, str or tuple, or matplotlib.colors.Colormap
        Input colors to display. Can be:
        - A colormap name or instance
        - A single color str or rgb tuple
        - A list of colors str or rgb tuples
    factor : float, default=0.2
        Increment in saturation between 0 and 1:
    keep_alpha : bool, default=False
        If True, keep the alpha channel
    mode : str, default='hls'
        If 'hls' or 'hsl', use HLS color space to saturate the colors
        If 'hsv', use HSV color space to saturate the colors

    Returns
    -------
    list of str or tuple, str or tuple, or matplotlib.colors.Colormap
        Saturated version of the input color or colors.
        Returns same type as input.

    Examples
    --------
    >>> saturate('#FF0000', factor=0.5)  # Saturate red
    >>> saturate(['#FF0000', '#00FF00'], factor=0.3)  # Saturate multiple colors
    >>> saturate('viridis', factor=0.3)  # Saturate entire colormap
    """
    if mode not in ['hls', 'hsl', 'hsv']:
        raise ValueError("mode must be 'hls' or 'hsv'")
    factor = np.clip(factor, 0, 1)
    try:
        if _is_cmap(c):
            c = plt.get_cmap(c)
            colors = get_colors(c, exclude_extreme=False)
            if not _is_qualitative(c):
                colors = get_colors(c, 256, exclude_extreme=False)
            category = 'cmap'
        elif _is_arraylike(c) and not _is_rgb(c):
            colors = c
            category = 'hex' if isinstance(c[0], str) else 'rgb'
        else:
            colors = [c]
            category = 'hex' if isinstance(c, str) else 'rgb'
    except:
        colors = None
        category = None
    if colors is not None and len(colors) > 1:
        colors = [saturate(color, factor, keep_alpha, mode) for color in colors]
        if category == 'cmap':
            name = c.name + "_saturated"
            if _is_qualitative(c):
                colors = ListedColormap(colors, name=name)
            else:
                colors = LinearSegmentedColormap.from_list(name, colors)
    else:
        try:
            if mode == 'hsv':
                hsv = np.array(to_hsv(colors[0], keep_alpha=keep_alpha))
                hsv[1] = np.clip(hsv[1] + factor, 0, 1)
                rgb = mc.hsv_to_rgb(hsv[:3])
                rgb = rgb + (hsv[3],) if hsv.shape[0] == 4 else rgb
                rgb = tuple([float(np.round(x, 3)) for x in rgb])
            elif mode in ['hls', 'hsl']:
                hls = np.array(to_hls(colors[0], keep_alpha=keep_alpha))
                hls[2] = np.clip(hls[2] + factor, 0, 1)
                rgb = colorsys.hls_to_rgb(*hls[:3])
                rgb = rgb + (hls[3],) if hls.shape[0] == 4 else rgb
            if category == 'hex':
                colors = to_hex(rgb, keep_alpha=keep_alpha).upper()
            else:
                colors = to_rgb(rgb, keep_alpha=keep_alpha)
        except:
            colors = None
    return colors


def desaturate(
    c: Union[Colormap, str, List[str], Tuple[float, ...], List[Tuple[float, ...]]],
    factor: float = 0.1,
    keep_alpha: bool = False,
    mode: str = 'hls'
) -> Union[str, List[str], Tuple[float, ...], List[Tuple[float, ...]]]:
    """
    Desaturates colors or a colormap by adjusting their saturation.

    This function takes colors or a colormap and returns less saturated versions by
    decreasing their saturation in HLS color space. 

    Parameters
    ----------
    c : list of str or tuple, str or tuple, or matplotlib.colors.Colormap
        Input colors to display. Can be:
        - A colormap name or instance
        - A single color str or rgb tuple
        - A list of colors str or rgb tuples
    factor : float, default=0.2
        Decrement in saturation between 0 and 1:
    keep_alpha : bool, default=False
        If True, keep the alpha channel
    mode : str, default='hls'
        If 'hls' or 'hsl', use HLS color space to desaturate the colors
        If 'hsv', use HSV color space to desaturate the colors

    Returns
    -------
    list of str or tuple, str or tuple, or matplotlib.colors.Colormap
        Desaturated version of the input color or colors.
        Returns same type as input.

    Examples
    --------
    >>> desaturate('#FF0000', factor=0.5)  # Desaturate red
    >>> desaturate(['#FF0000', '#00FF00'], factor=0.3)  # Desaturate multiple colors
    >>> desaturate('viridis', factor=0.3)  # Desaturate entire colormap
    """
    if mode not in ['hls', 'hsl', 'hsv']:
        raise ValueError("mode must be 'hls' or 'hsv'")
    factor = np.clip(factor, 0, 1)
    try:
        if _is_cmap(c):
            c = plt.get_cmap(c)
            colors = get_colors(c, exclude_extreme=False)
            if not _is_qualitative(c):
                colors = get_colors(c, 256, exclude_extreme=False)
            category = 'cmap'
        elif _is_arraylike(c) and not _is_rgb(c):
            colors = c
            category = 'hex' if isinstance(c[0], str) else 'rgb'
        else:
            colors = [c]
            category = 'hex' if isinstance(c, str) else 'rgb'
    except:
        colors = None
        category = None
    if colors is not None and len(colors) > 1:
        colors = [desaturate(color, factor, keep_alpha, mode) for color in colors]
        if category == 'cmap':
            name = c.name + "_desaturated"
            if _is_qualitative(c):
                colors = ListedColormap(colors, name=name)
            else:
                colors = LinearSegmentedColormap.from_list(name, colors)
    else:
        try:
            if mode == 'hsv':
                hsv = np.array(to_hsv(colors[0], keep_alpha=keep_alpha))
                hsv[1] = np.clip(hsv[1] - factor, 0, 1)
                rgb = mc.hsv_to_rgb(hsv[:3])
                rgb = rgb + (hsv[3],) if hsv.shape[0] == 4 else rgb
                rgb = tuple([float(np.round(x, 3)) for x in rgb])
            elif mode in ['hls', 'hsl']:
                hls = np.array(to_hls(colors[0], keep_alpha=keep_alpha))
                hls[2] = np.clip(hls[2] - factor, 0, 1)
                rgb = colorsys.hls_to_rgb(*hls[:3])
                rgb = rgb + (hls[3],) if hls.shape[0] == 4 else rgb
            if category == 'hex':
                colors = to_hex(rgb, keep_alpha=keep_alpha).upper()
            else:
                colors = to_rgb(rgb, keep_alpha=keep_alpha)
        except:
            colors = None
    return colors


def to_hex(c: Union[str, Tuple[float, ...]], keep_alpha: bool = False, safe: bool = False) -> str:
    """
    Converts a color to its hexadecimal representation.

    This function takes a color in various formats (name, RGB, RGBA) and converts
    it to a hexadecimal string representation. The alpha channel can be optionally
    included in the output.

    Parameters
    ----------
    c : str or tuple of float
        Input color. Can be:
        - A color name (e.g., 'red')
        - A hex string (e.g., '#FF0000')
        - An RGB tuple (e.g., (1.0, 0.0, 0.0))
        - An RGBA tuple (e.g., (1.0, 0.0, 0.0, 1.0))
    keep_alpha : bool, default=False
        If True, includes the alpha channel in the hex string
        If False, returns a 6-character hex string without alpha
    safe : bool, default=False
        If True, converts only 0.0 - 1.0 RGB range
        If False, converts both 0.0 - 1.0 and 0 - 255 RGB ranges

    Returns
    -------
    str or None
        Hexadecimal color string:
        - '#RRGGBB' if keep_alpha is False
        - '#RRGGBBAA' if keep_alpha is True
        - None if conversion fails

    Examples
    --------
    >>> to_hex('red')
    '#FF0000'
    >>> to_hex((1.0, 0.0, 0.0, 0.5), keep_alpha=True)
    '#FF000080'
    """
    hexname = None
    try:
        if isinstance(c, tuple) and max(c) > 1.0 and not safe:
            c = tuple([c_ / 255.0 for c_ in c])
        hexname = mc.to_hex(c, keep_alpha).upper()
    except:
        hexname = None
    return hexname


def to_rgb(c: Union[str, Tuple[float, ...]], keep_alpha: bool = False, fmt: int = 1, safe: bool = False) -> Optional[Tuple[float, ...]]:
    """
    Converts a color to its RGB (Red, Green, Blue) representation.

    This function takes a color in various formats (name, hex, RGB, RGBA) and
    converts it to an RGB tuple with values in the range [0, 1]. If the input
    cannot be converted to a valid color, returns None.

    Parameters
    ----------
    c : str or tuple of float
        Input color. Can be:
        - A color name (e.g., 'red')
        - A hex string (e.g., '#FF0000')
        - An RGB tuple (e.g., (1.0, 0.0, 0.0))
        - An RGBA tuple (e.g., (1.0, 0.0, 0.0, 1.0))
    keep_alpha : bool, default=False
        If True, includes the alpha channel in the RGB tuple
        If False, returns a 3-character RGB tuple without alpha
    fmt : int, default=1
        Format of the output RGB tuple (1 or 255):
        - 1: 0.0-1.0 RGB range
        - 255: 0-255 RGB range
    safe : bool, default=False
        If True, converts only 0.0 - 1.0 RGB range
        If False, converts both 0.0 - 1.0 and 0 - 255 RGB ranges

    Returns
    -------
    tuple of float or None
        RGB tuple with values in range [0, 1], or None if conversion fails.
        The tuple format is (red, green, blue) where each component is a float
        between 0 and 1.

    Examples
    --------
    >>> to_rgb('red')
    (1.0, 0.0, 0.0)
    >>> to_rgb('#00FF00')
    (0.0, 1.0, 0.0)
    >>> to_rgb((0.0, 0.0, 1.0))
    (0.0, 0.0, 1.0)
    """
    rgb = None
    try:
        color = c
        if _is_rgb(c):
            if max(c) > 1.0 and not safe:
                color = tuple([np.round(c_ / 255.0, 3) for c_ in c])
            else:
                color = tuple([np.round(c_, 3) for c_ in c])
        if keep_alpha:
            rgb = mc.to_rgba(color)
        else:
            rgb = mc.to_rgb(color)
        rgb = tuple([float(np.round(c_, 3)) for c_ in rgb])
    except:
        rgb = None
    if fmt not in [1, 255]:
        raise ValueError(f"Invalid format: {fmt}")  
    if fmt == 255 and rgb is not None:
        rgb = tuple([int(np.round(c_ * 255)) for c_ in rgb])
    return rgb


def to_hls(c: Union[str, Tuple[float, ...]], keep_alpha: bool = False, fmt: int = 1, safe: bool = False) -> Optional[Tuple[float, ...]]:
    """
    Converts a color to its HLS (Hue, Lightness, Saturation) representation.

    This function takes a color in various formats (name, hex, RGB, RGBA) and
    converts it to an HSL tuple with values in the range [0, 1]. If the input
    cannot be converted to a valid color, returns None.

    Parameters
    ----------
    c : str or tuple of float
        Input color. Can be:
        - A color name (e.g., 'red')
        - A hex string (e.g., '#FF0000')
        - An RGB tuple (e.g., (1.0, 0.0, 0.0))
        - An RGBA tuple (e.g., (1.0, 0.0, 0.0, 1.0))
    keep_alpha : bool, default=False
        If True, includes the alpha channel in the HLS tuple
        If False, returns a 3-character HLS tuple without alpha
    fmt : int, default=1
        Format of the output HLS tuple (1 or 255):
        - 1: 0.0-1.0 HLS range
        - 255: 0-255 HLS range
    safe : bool, default=False
        If True, converts only 0.0 - 1.0 HLS range
        If False, converts both 0.0 - 1.0 and 0 - 255 HLS ranges

    Returns
    -------
    tuple of float or None
        HLS tuple with values in range [0, 1], or None if conversion fails.
        The tuple format is (hue, lightness, saturation) where each component is a float
        between 0 and 1.

    Examples
    --------
    >>> to_hls('red')
    (0.0, 0.5, 1.0)
    >>> to_hls('#00FF00')
    (0.0, 0.5, 1.0)
    >>> to_hls((0.0, 0.0, 1.0))
    (0.0, 0.5, 1.0)
    """
    hls = None
    try:
        rgb = to_rgb(c, keep_alpha=keep_alpha, fmt=1, safe=safe)
        hls = colorsys.rgb_to_hls(*rgb[:3])
        hls = tuple([float(np.round(c_, 3)) for c_ in hls])
        if len(rgb) == 4:
            hls = hls + (rgb[3],)
    except:
        hls = None
    if fmt not in [1, 255]:
        raise ValueError(f"Invalid format: {fmt}")  
    if fmt == 255 and hls is not None:
        hls = tuple([int(np.round(c_ * 255)) for c_ in hls])
    return hls


def to_hsv(c: Union[str, Tuple[float, ...]], keep_alpha: bool = False, fmt: int = 1, safe: bool = False) -> Optional[Tuple[float, ...]]:
    """
    Converts a color to its HSV (Hue, Saturation, Value) representation.

    This function takes a color in various formats (name, hex, RGB, RGBA) and
    converts it to an HSV tuple with values in the range [0, 1]. If the input
    cannot be converted to a valid color, returns None.

    Parameters
    ----------
    c : str or tuple of float
        Input color. Can be:
        - A color name (e.g., 'red')
        - A hex string (e.g., '#FF0000')
        - An RGB tuple (e.g., (1.0, 0.0, 0.0))
        - An RGBA tuple (e.g., (1.0, 0.0, 0.0, 1.0))
    keep_alpha : bool, default=False
        If True, includes the alpha channel in the HSV tuple
        If False, returns a 3-character HSV tuple without alpha
    fmt : int, default=1
        Format of the output HSV tuple (1 or 255):
        - 1: 0.0-1.0 HSV range
        - 255: 0-255 HSV range
    safe : bool, default=False
        If True, converts only 0.0 - 1.0 HSV range
        If False, converts both 0.0 - 1.0 and 0 - 255 HSV ranges

    Returns
    -------
    tuple of float or None
        HSV tuple with values in range [0, 1], or None if conversion fails.
        The tuple format is (hue, saturation, value) where each component is a float
        between 0 and 1.

    Examples
    --------
    >>> to_hsv('red')
    (0.0, 0.5, 1.0)
    >>> to_hsv('#00FF00')
    (0.0, 0.5, 1.0)
    >>> to_hsv((0.0, 0.0, 1.0))
    (0.0, 0.5, 1.0)
    """
    hsv = None
    try:
        rgb = to_rgb(c, keep_alpha=keep_alpha, fmt=1, safe=safe)
        hsv = mc.rgb_to_hsv(rgb[:3])
        hsv = tuple([float(np.round(c_, 3)) for c_ in hsv])
        if len(rgb) == 4:
            hsv = hsv + (rgb[3],)
    except:
        hsv = None
    if fmt not in [1, 255]:
        raise ValueError(f"Invalid format: {fmt}")  
    if fmt == 255 and hsv is not None:
        hsv = tuple([int(np.round(c_ * 255)) for c_ in hsv])
    return hsv


import types
__all__ = [name for name, thing in globals().items()
          if not (name.startswith("_") or isinstance(thing, types.ModuleType))]
del types


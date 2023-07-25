#!/usr/bin/env python
# -*- coding: utf8 -*-

import numpy as np


def rotate2D(x, y, phi):
    """
    Rotates X and Y coordinates in 2D plane
    
    Parameters
    ----------
    x : ndarray
        X coordinates
    y : ndarray
        Y coordinates
    phi : float
        Angle [radians]

    Returns
    -------
    x_ : ndarray
        Rotated X coordinates
    y_ : ndarray
        Rotated Y coordinates

    """
    x = np.asarray(x)
    y = np.asarray(y)
    r = np.stack([[np.cos(phi), -np.sin(phi)],
                  [np.sin(phi), np.cos(phi)]])
    x_, y_ = np.dot(r, np.stack([x, y]))
    return x_, y_


def get_circle_dots(r=1, n=360):
    """
    Returns poits at equal arc length on a circle
    
    Parameters
    ----------
    r : float, default 1
        Radius

    Returns
    -------
    x : ndarray
        X coordinates
    y : ndarray
        Y coordinates

    """
    phi = np.linspace(0, 2 * np.pi, n + 1)[:-1]
    z = r * np.exp(1j * phi)
    x, y = z.real, z.imag
    return x, y
    

def get_ellipse_dots(a=5, b=3, n=360):
    """
    Returns poits at equal arc length on an ellipse
    
    Parameters
    ----------
    a : float, default 5
        Semi-major axis
    b : float, default 3
        Semi-minor axis
    n : int, default 360
        Number of points

    Returns
    -------
    x : ndarray
        X coordinates
    y : ndarray
        Y coordinates

    """
    # Initialize half of points with increased density
    x = np.linspace(a, -a, 20 * n)
    y = np.sqrt(a**2 - x**2) * b / a
    # Add lower half of points
    x = np.concatenate([x[:-1], x[::-1]])
    y = np.concatenate([y[:-1], -y[::-1]])
    # Calc distance between high-density points
    l = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    # Calc distance between requested points
    dl = np.sum(l) / n
    # Find indices of equidistant points along the ellipse arc
    l = np.cumsum(l) % dl
    d = np.diff(l)
    mask = (d[1:-1] < d[:-2]) & (d[1:-1] < d[2:])
    idx = np.arange(len(mask)) + 2
    idx = np.concatenate([(0,), idx[mask]])
    x = x[idx]
    y = y[idx]
    return x, y


def distort_radius(x, y, p):
    """
    Distorts radius by adding noise p
    
    Parameters
    ----------
    x : ndarray
        X coordinates centered at zero
    y : ndarray
        Y coordinates centered at zero
    p : ndarray
        Noise

    Returns
    -------
    x : ndarray
        X coordinates with distorted radius to zero
    y : ndarray
        Y coordinates with distorted radius to zero

    """
    z = x + 1j * y
    phi = np.angle(z)
    r = np.abs(z) + p
    z = r * np.exp(1j * phi)
    x, y = z.real, z.imag
    return x, y




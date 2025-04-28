import numpy as np
import pytest
from matplotlib.colors import Colormap, LinearSegmentedColormap
from PIL import Image

from excolor.colortools import (_is_cmap, _is_arraylike, _is_rgb, _aspect_ratio,
                              show_colors, hex_to_rgb, rgb_to_hex, get_colors,
                              get_colors_rgb, get_colors_hex)

def test_is_cmap():
    # Test with string input
    assert _is_cmap('viridis') == True
    assert _is_cmap('not_a_cmap') == False
    
    # Test with Colormap input
    cmap = LinearSegmentedColormap.from_list('custom', ['red', 'blue'])
    assert _is_cmap(cmap) == True
    
    # Test with invalid input types
    assert _is_cmap(123) == False
    assert _is_cmap([1, 2, 3]) == False
    assert _is_cmap(None) == False

def test_is_arraylike():
    # Test with various array-like objects
    assert _is_arraylike(np.array([1, 2, 3])) == True
    assert _is_arraylike([1, 2, 3]) == True
    assert _is_arraylike((1, 2, 3)) == True
    assert _is_arraylike({1, 2, 3}) == True

    # Test with non-array-like objects
    assert _is_arraylike({1: 'A'}) == False
    assert _is_arraylike("string") == False
    assert _is_arraylike(123) == False
    assert _is_arraylike(None) == False

def test_is_rgb():
    # Test valid RGB arrays
    assert _is_rgb(np.array([0.5, 0.5, 0.5])) == True
    assert _is_rgb(np.array([0.5, 0.5, 0.5, 1.0])) == True  # RGBA
    
    # Test invalid RGB arrays
    assert _is_rgb(np.array([0.5, 0.5])) == False  # Too few components
    assert _is_rgb(np.array([0.5, 0.5, 0.5, 1.0, 1.0])) == False  # Too many components
    assert _is_rgb(np.array([2.0, 0.5, 0.5])) == False  # Values out of range
    assert _is_rgb(np.array([-0.1, 0.5, 0.5])) == False  # Values out of range

def test_aspect_ratio():
    # Test various input lengths
    assert _aspect_ratio(12) == (4, 3)
    assert _aspect_ratio(16) == (4, 4)
    assert _aspect_ratio(20) == (5, 4)
    
    # Test with minimum length constraint
    assert _aspect_ratio(12, lmin=5) == (6, 2)
    
    # Test edge cases
    assert _aspect_ratio(1) == (1, 1)
    assert _aspect_ratio(2) == (2, 1)

def test_hex_to_rgb():
    # Test valid hex colors
    assert np.allclose(hex_to_rgb('#FF0000'), [1.0, 0.0, 0.0])
    assert np.allclose(hex_to_rgb('#00FF00'), [0.0, 1.0, 0.0])
    assert np.allclose(hex_to_rgb('#0000FF'), [0.0, 0.0, 1.0])
    
    # Test shorthand hex
    assert np.allclose(hex_to_rgb('#F00'), [1.0, 0.0, 0.0])
    
    # Test with/without hash
    assert np.allclose(hex_to_rgb('FF0000'), [1.0, 0.0, 0.0])
    
    # Test invalid hex colors
    with pytest.raises(ValueError):
        hex_to_rgb('invalid')
    with pytest.raises(ValueError):
        hex_to_rgb('#GG0000')

def test_rgb_to_hex():
    # Test RGB arrays
    assert rgb_to_hex([1.0, 0.0, 0.0]) == '#ff0000'
    assert rgb_to_hex([0.0, 1.0, 0.0]) == '#00ff00'
    assert rgb_to_hex([0.0, 0.0, 1.0]) == '#0000ff'
    
    # Test with numpy arrays
    assert rgb_to_hex(np.array([1.0, 0.0, 0.0])) == '#ff0000'
    
    # Test with invalid inputs
    with pytest.raises(ValueError):
        rgb_to_hex([2.0, 0.0, 0.0])  # Values > 1
    with pytest.raises(ValueError):
        rgb_to_hex([-0.1, 0.0, 0.0])  # Values < 0

def test_get_colors():
    # Test with valid inputs
    n = 5
    colors = get_colors(n, cmap='viridis')
    assert len(colors) == n
    assert all(isinstance(c, np.ndarray) for c in colors)
    
    # Test with custom colormap
    custom_cmap = LinearSegmentedColormap.from_list('custom', ['red', 'blue'])
    colors = get_colors(n, cmap=custom_cmap)
    assert len(colors) == n
    
    # Test invalid inputs
    with pytest.raises(ValueError):
        get_colors(0)  # n must be positive
    with pytest.raises(ValueError):
        get_colors(n, cmap='invalid_cmap')

def test_get_colors_rgb():
    # Test with valid inputs
    n = 5
    colors = get_colors_rgb(n, cmap='viridis')
    assert len(colors) == n
    assert all(isinstance(c, np.ndarray) for c in colors)
    assert all(len(c) == 3 for c in colors)  # RGB arrays
    
    # Test value ranges
    assert all(np.all((c >= 0) & (c <= 1)) for c in colors)

def test_get_colors_hex():
    # Test with valid inputs
    n = 5
    colors = get_colors_hex(n, cmap='viridis')
    assert len(colors) == n
    assert all(isinstance(c, str) for c in colors)
    assert all(c.startswith('#') for c in colors)
    assert all(len(c) == 7 for c in colors)  # #RRGGBB format

def test_show_colors():
    # Test basic functionality (no error)
    colors = ['#FF0000', '#00FF00', '#0000FF']
    show_colors(colors)
    
    # Test with different input types
    show_colors(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    show_colors(['red', 'green', 'blue'])
    
    # Test with custom figsize and title
    show_colors(colors, figsize=(8, 2), title='Test Colors') 
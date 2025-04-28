import numpy as np
import pytest
from PIL import Image

from excolor.imagetools import (load_image, show_image, image_to_array,
                              array_to_image, smoother, find_peaks, get_mask)

@pytest.fixture
def sample_image():
    # Create a simple test image
    img = Image.new('RGB', (100, 100), color='white')
    return img

@pytest.fixture
def sample_array():
    # Create a sample numpy array representing an image
    return np.ones((100, 100, 3), dtype=np.float32)

def test_load_image(tmp_path):
    # Create a temporary test image
    test_img_path = tmp_path / "test.png"
    img = Image.new('RGB', (100, 100), color='white')
    img.save(test_img_path)
    
    # Test loading from path
    loaded_img = load_image(str(test_img_path))
    assert isinstance(loaded_img, Image.Image)
    assert loaded_img.size == (100, 100)
    assert loaded_img.mode == 'RGB'
    
    # Test loading from PIL Image
    loaded_img2 = load_image(img)
    assert isinstance(loaded_img2, Image.Image)
    assert loaded_img2.size == (100, 100)
    
    # Test loading from numpy array
    arr = np.array(img)
    loaded_img3 = load_image(arr)
    assert isinstance(loaded_img3, Image.Image)
    assert loaded_img3.size == (100, 100)
    
    # Test invalid input
    with pytest.raises(ValueError):
        load_image("nonexistent.png")
    with pytest.raises(ValueError):
        load_image(123)

def test_show_image(sample_image):
    # Test basic functionality (no error)
    show_image(sample_image)
    
    # Test with numpy array
    arr = np.array(sample_image)
    show_image(arr)
    
    # Test with custom figsize and title
    show_image(sample_image, figsize=(8, 8), title='Test Image')

def test_image_to_array(sample_image):
    # Test conversion from PIL Image
    arr = image_to_array(sample_image)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (100, 100, 3)
    assert arr.dtype == np.float32
    assert np.all((arr >= 0) & (arr <= 1))
    
    # Test with grayscale image
    gray_img = sample_image.convert('L')
    arr_gray = image_to_array(gray_img)
    assert arr_gray.shape == (100, 100)
    
    # Test with RGBA image
    rgba_img = sample_image.convert('RGBA')
    arr_rgba = image_to_array(rgba_img)
    assert arr_rgba.shape == (100, 100, 4)

def test_array_to_image(sample_array):
    # Test conversion from numpy array
    img = array_to_image(sample_array)
    assert isinstance(img, Image.Image)
    assert img.size == (100, 100)
    assert img.mode == 'RGB'
    
    # Test with grayscale array
    gray_arr = np.ones((100, 100), dtype=np.float32)
    img_gray = array_to_image(gray_arr)
    assert img_gray.mode == 'L'
    
    # Test with RGBA array
    rgba_arr = np.ones((100, 100, 4), dtype=np.float32)
    img_rgba = array_to_image(rgba_arr)
    assert img_rgba.mode == 'RGBA'
    
    # Test invalid inputs
    with pytest.raises(ValueError):
        array_to_image(np.ones((100,)))  # Wrong dimensions
    with pytest.raises(ValueError):
        array_to_image(np.ones((100, 100, 5)))  # Too many channels

def test_smoother():
    # Create test data
    x = np.linspace(0, 10, 100)
    y = np.sin(x) + np.random.normal(0, 0.1, 100)
    
    # Test with default parameters
    smoothed = smoother(y)
    assert len(smoothed) == len(y)
    assert np.all(np.abs(smoothed) <= np.abs(y).max())
    
    # Test with custom window size
    smoothed_custom = smoother(y, window=15)
    assert len(smoothed_custom) == len(y)
    
    # Test with invalid inputs
    with pytest.raises(ValueError):
        smoother([])  # Empty array
    with pytest.raises(ValueError):
        smoother(y, window=0)  # Invalid window size

def test_find_peaks():
    # Create test data with known peaks
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    # Test peak finding
    peaks = find_peaks(y)
    assert isinstance(peaks, np.ndarray)
    assert len(peaks) > 0
    assert all(isinstance(p, np.integer) for p in peaks)
    
    # Test with threshold
    peaks_threshold = find_peaks(y, threshold=0.8)
    assert len(peaks_threshold) <= len(peaks)
    
    # Test with invalid inputs
    with pytest.raises(ValueError):
        find_peaks([])  # Empty array
    with pytest.raises(ValueError):
        find_peaks(y, threshold=-1)  # Invalid threshold

def test_get_mask(sample_image):
    # Test basic mask creation
    mask = get_mask(sample_image)
    assert isinstance(mask, np.ndarray)
    assert mask.shape == (100, 100)
    assert mask.dtype == bool
    
    # Test with threshold
    mask_threshold = get_mask(sample_image, threshold=0.8)
    assert isinstance(mask_threshold, np.ndarray)
    assert mask_threshold.shape == (100, 100)
    
    # Test with different image types
    gray_img = sample_image.convert('L')
    mask_gray = get_mask(gray_img)
    assert mask_gray.shape == (100, 100)
    
    rgba_img = sample_image.convert('RGBA')
    mask_rgba = get_mask(rgba_img)
    assert mask_rgba.shape == (100, 100)
    
    # Test with numpy array input
    arr = np.array(sample_image)
    mask_arr = get_mask(arr)
    assert mask_arr.shape == (100, 100) 
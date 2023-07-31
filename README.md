[![Python Versions](https://img.shields.io/pypi/pyversions/excolor?style=plastic)](https://pypi.org/project/excolor/)
[![PyPI](https://img.shields.io/pypi/v/excolor?style=plastic)](https://pypi.org/project/excolor/)
[![License](https://img.shields.io/pypi/l/excolor?style=plastic)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/excolor/badge/?version=latest)](https://excolor.readthedocs.io/en/latest/?badge=latest)

# excolors

## Extexnded color utilities for python
#

# Installation
```
pip install excolor
```


# Extra colormaps

```
import excolor

cmap = plt.get_cmap("gruvbox")
cmap
```

![](https://github.com/timpyrkov/excolor/blob/master/img/colormap.png?raw=true)

# Colorize black-and-white image

- Hue and saturation cannot colorize black-and-white image. They only change pixels where levels of red, gree, blue are not the same. 

- Colorize() function takes a greyscale or b&w image and adds colors to dark and light areas.


```
fname = "pacman.png"

img = excolor.load_image(fname)

plt.figure(figsize=(2,2), facecolor="#00000000")
plt.imshow(img)
plt.axis("off")
plt.show()

img = excolor.colorize(fname, "yellow", "green")

plt.figure(figsize=(2,2), facecolor="#00000000")
plt.imshow(img)
plt.axis("off")
plt.show()

```

![](https://github.com/timpyrkov/excolor/blob/master/img/pacman.png?raw=true)
![](https://github.com/timpyrkov/excolor/blob/master/img/arrow.png?raw=true)
![](https://github.com/timpyrkov/excolor/blob/master/img/colorized.png?raw=true)

# Color cycler

Set ax color cycler based on cmap or list of colors

```
from sklearn.metrics import confusion_matrix

# Generate data for heatmap and scatter plot
np.random.seed(10)
y_true = (np.arange(50) > 25).astype(float)
x0 = np.random.normal(0,1,50) + y_true
x1 = np.random.normal(0,1,50) + y_true

# Predict y_true based on x0
y_pred = x0 > 0
c = confusion_matrix(y_true, y_pred)

# Plot heatmap of confusion matrix using "viridis" colormap
plt.figure(figsize=(4,2), facecolor="white")
plt.subplot(121)
plt.title("Heatmap")
sns.heatmap(c, annot=True, cmap="viridis", vmin=0)
plt.subplot(122)
plt.title("Scatterplot")

# Set "viridis" colors for color cycler to make scatter plot
excolor.set_color_cycler("viridis", n=2)
for i in range(2):
    mask = y_true == i
    plt.scatter(x0[mask], x1[mask])
plt.tight_layout()
plt.show()


```

![](https://github.com/timpyrkov/excolor/blob/master/img/charts.png?raw=true)

# Log-scaling colormaps

Color perlin noise using log-scaled colormap to visualize water caustics

```
from pythonperlin import perlin

dens = 32
shape = (8,8)
x = perlin(shape, dens=dens)

# Log-scaled colormap
cmap = excolor.logscale_cmap("GnBu_r")

plt.figure(figsize=(6,6), facecolor="#00000000")
plt.imshow( np.abs(x), cmap=cmap)
plt.axis("off")
plt.show()
```

![](https://github.com/timpyrkov/excolor/blob/master/img/caustics.png?raw=true)

# Documentation

[https://excolor.readthedocs.io](https://excolor.readthedocs.io)
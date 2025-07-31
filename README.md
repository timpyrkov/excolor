[![Python Versions](https://img.shields.io/pypi/pyversions/excolor?style=plastic)](https://pypi.org/project/excolor/)
[![PyPI](https://img.shields.io/pypi/v/excolor?style=plastic)](https://pypi.org/project/excolor/)
[![License](https://img.shields.io/pypi/l/excolor?style=plastic)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/excolor/badge/?version=latest)](https://excolor.readthedocs.io/en/latest/?badge=latest)


<h1><p align="left">
  <img src="https://github.com/timpyrkov/excolor/blob/master/img/logo.png?raw=true" alt="ExColor logo" height="30" style="vertical-align: middle; margin-right: 10px;">
  <span style="font-size:2.5em; vertical-align: middle;"><b>ExColor</b></span>
</p></h1>


# Installation

```
pip install excolor
```


# Extra colormaps (gruvbox and more)

New colormaps: BrBu, BrGn, OrBu, OrGn, PiBu, rtd, artdeco, cyberpunk, synthwave, gruvbox, cobalt, noctis, monokai, oceanic

```
import excolor

cmap = plt.get_cmap("gruvbox")
cmap
```

`Gruvbox`
![](https://github.com/timpyrkov/excolor/blob/master/img/colormap.png?raw=true)

# Generate color palettes

Mode "superellipse" defines the path from black to white. Higher degree produces more vibrant colors.

```
# Seed color
color = '#458588'

colors = excolor.generate_palette(color, mode='superellipse', power=3)

excolor.show_colors(colors)
```

`['#FFFFFF', '#CFF4F6', '#A7DBDE', '#84C0C3', '#63A3A6', '#458588', '#296568', '#114547', '#002224', '#000000']`
![](https://github.com/timpyrkov/excolor/blob/master/img/palette.png?raw=true)

```
# Generate palette for primary CSS colors
primary_colors = excolor.generate_primary_palette(color)

excolor.show_colors(primary_colors)
```

`/* Kumutoto color palette */`<br>
`:root {`<br>
`  --primary-color-1: #DBFFFF;`<br>
`  --primary-color-2: #BBF0F3;`<br>
`  --primary-color-3: #A0DCDF;`<br>
`  --primary-color-4: #88C6C9;`<br>
`  --primary-color-5: #70B0B3;`<br>
`  --primary-color-6: #599A9C;`<br>
`  --primary-color-7: #438386;`<br>
`  --primary-color-8: #2D6C6F;`<br>
`  --primary-color-9: #175558;`<br>
`}`<br>
![](https://github.com/timpyrkov/excolor/blob/master/img/palette_primary.png?raw=true)


```
# Generate palette for background CSS colors
background_colors = excolor.generate_background_palette(color)

excolor.show_colors(background_colors)
```

`/* Cyprus color palette */`<br>
`:root {`<br>
`  --background-color-1: #2D6C6F;`<br>
`  --background-color-2: #226063;`<br>
`  --background-color-3: #175558;`<br>
`  --background-color-4: #0D494C;`<br>
`  --background-color-5: #043D3F;`<br>
`  --background-color-6: #003033;`<br>
`  --background-color-7: #002426;`<br>
`  --background-color-8: #001618;`<br>
`  --background-color-9: #00090A;`<br>
`}`<br>
![](https://github.com/timpyrkov/excolor/blob/master/img/palette_background.png?raw=true)


```
# Generate palette for primary CSS colors
foreground_colors = excolor.generate_foreground_palette(color)

excolor.show_colors(foreground_colors)
```

`/* Blizzard Blue color palette */`<br>
`:root {`<br>
`  --foreground-color-1: #EDFFFF;`<br>
`  --foreground-color-2: #DBFFFF;`<br>
`  --foreground-color-3: #CAFAFC;`<br>
`  --foreground-color-4: #BBF0F3;`<br>
`  --foreground-color-5: #ADE6E9;`<br>
`  --foreground-color-6: #A0DCDF;`<br>
`  --foreground-color-7: #94D1D4;`<br>
`  --foreground-color-8: #88C6C9;`<br>
`  --foreground-color-9: #7CBBBE;`<br>
`}`<br>
![](https://github.com/timpyrkov/excolor/blob/master/img/palette_foreground.png?raw=true)

# Colorize image

```
url = "https://github.com/timpyrkov/excolor/blob/master/img/image_bw.png?raw=true"

# Load image from url (image_bw.png from github repository)
img = excolor.load_image(url)

# Show source grayscale image
plt.figure(figsize=(6,6), facecolor="#00000000")
plt.imshow(img)
plt.axis("off")
plt.show()

# Colorize and show image
green, blue = "#CAC94C", "#226063"
img = excolor.colorize_image(url, green, blue)

# Show colorized image
plt.figure(figsize=(6,6), facecolor="#00000000")
plt.imshow(img)
plt.axis("off")
plt.show()
```

![](https://github.com/timpyrkov/excolor/blob/master/img/image_bw.png?raw=true)
![](https://github.com/timpyrkov/excolor/blob/master/img/image_color.png?raw=true)
![](https://github.com/timpyrkov/excolor/blob/master/img/colorized.png?raw=true)


# Greyscale image, keep channels

```
url = "https://github.com/timpyrkov/excolor/blob/master/img/image_color.png?raw=true"

# Load image from url (image_color.png from github repository)
img = excolor.load_image(url)

# Show source image
plt.figure(figsize=(6,6), facecolor="#00000000")
plt.imshow(img)
plt.axis("off")
plt.show()

# Convert to greyscale
greyscaled = excolor.greyscale_image(url)

# Show colorized image
plt.figure(figsize=(6,6), facecolor="#00000000")
plt.imshow(greyscaled)
plt.axis("off")
plt.show()

print("Source image array shape:", np.asarray(img).shape)
print("Greyscaled image array shape:", np.asarray(greyscaled).shape)
```

![](https://github.com/timpyrkov/excolor/blob/master/img/image_bw.png?raw=true)
![](https://github.com/timpyrkov/excolor/blob/master/img/image_color.png?raw=true)
![](https://github.com/timpyrkov/excolor/blob/master/img/colorized.png?raw=true)


# Low-polygonal

```
url = "https://github.com/timpyrkov/excolor/blob/master/img/image_gruvbox.png?raw=true"

# Load image from url (image_color.png from github repository)
img = excolor.load_image(url)

# Show source image
plt.figure(figsize=(6,6), facecolor="#00000000")
plt.imshow(img)
plt.axis("off")
plt.show()

# Convert to low-polygonal image
img = excolor.triangle_wallpaper(img=img, density=30, distortion=0.4, size=(640,480))

# Sjow low-polygonal image
plt.figure(figsize=(6,6), facecolor="#00000000")
plt.imshow(img)
plt.axis("off")
plt.show()
```

![](https://github.com/timpyrkov/excolor/blob/master/img/image_bw.png?raw=true)
![](https://github.com/timpyrkov/excolor/blob/master/img/image_color.png?raw=true)
![](https://github.com/timpyrkov/excolor/blob/master/img/colorized.png?raw=true)





```
# Seed color
color = '#70B3E0'

# Generate a palette for primary CSS colors
background_colors = excolor.generate_background_palette(color)

# Show primary colors
excolor.show_colors(background_colors)
```






# Draw gradient wallpaper



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
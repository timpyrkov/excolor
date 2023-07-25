# -*- coding: utf-8 -*-

from excolor.excolor import *
from excolor.utils import _add_extended_colormaps
from excolor.imagetools import *
from excolor.background import *
from excolor.geometry import *
from pkg_resources import get_distribution

_add_extended_colormaps()
del _add_extended_colormaps

__version__ = get_distribution("excolor").version
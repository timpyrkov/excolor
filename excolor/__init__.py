# -*- coding: utf-8 -*-

from excolor.excolor import *
from excolor.excolor import _add_extended_colormaps
from pkg_resources import get_distribution

_add_extended_colormaps()
del _add_extended_colormaps

__version__ = get_distribution("excolor").version
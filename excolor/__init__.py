# -*- coding: utf-8 -*-

from excolor.excolor import *
from pkg_resources import get_distribution

add_extended_colormaps()

__version__ = get_distribution("excolor").version
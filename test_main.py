import argparse
import queue

import numpy as np

from utils import FormFactor
from utils import TriangleSet, Triangle, distance
from utils.reader import XMLReader
from utils import XMLWriter
from utils import Isocell

def test():
    isocell = Isocell(rays=1000, div=5, isrand=0, draw_cells=True)

    print('testing!!!!')

    # print(isocell.XYr)
    # print(isocell.XYr)
    # print(isocell.A0)


if __name__ == '__main__':

    test()
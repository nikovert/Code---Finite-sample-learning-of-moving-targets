from math import*
import numpy as np
import random
from aeb import AEB

def prune(L, distance):


    assert L.size(1) == distance.size(), "x should be 'hello'"
    srtList =  L.sort()
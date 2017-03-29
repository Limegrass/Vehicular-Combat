"""
Utility functions used by the simulator modules.

Author: RR
"""
import csv
from collections import namedtuple

Position = namedtuple('Position', ['x', 'y'])

# modify the following path to match the folder to which you unzipped the
# contents
ROOT_PATH = 'E:\Davidson\AI\HW4\BicycleVehicleModel'
SIM_RESOLUTION = 0.01


class SimulationError(Exception):
    pass


def iterable_to_ctypes_array(data, output_type):
    """ Utility function to convert an iterable to a C array. """
    output_array = output_type()
    for i, d in enumerate(data):
        output_array[i] = d
    return output_array


def ctype_array_to_list(data):
    """ Utility function to convert a C array to a Python list. """
    output_list = list()
    for i in data:
        output_list.append(i)
    return output_list

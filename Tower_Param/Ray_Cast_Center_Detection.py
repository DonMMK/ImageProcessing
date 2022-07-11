from re import T
from PIL import Image, ImageDraw, ImageFilter
from PIL.ExifTags import TAGS, GPSTAGS
from matplotlib.transforms import Bbox
import numpy as np
import math 
import struct
import time
import matplotlib.pyplot as plt
import Tower_Parameter_Detection as TPD



if __name__ == "__main__":
    TPD.getAllframe_metaData()
    TPD.getframe_data()
    

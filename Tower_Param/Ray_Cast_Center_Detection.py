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



def getframe_data(frameNumber):

    if frameNumber < 759:
        frame_number = str(frameNumber)
        mask_frame_URL = "/home/don/Git/LargeDataFilesLinux/predictions-masks/image_frame_" + frame_number + ".tif"
    else:
        frame_number = str(frameNumber)
        mask_frame_URL = "/home/don/Git/LargeDataFilesLinux/predictions-masks_2/image_frame_" + frame_number + ".tif"
    
    # print(mask_frame_URL)
    mask_image = Image.open(mask_frame_URL) 

    # mask_image.show()

    pixeldata = mask_image.getdata()
    
    reshape_data = np.reshape(pixeldata, (mask_image.height, mask_image.width))
    
    # print("Done Get Frame Data")
    return reshape_data

def getAllframe_metaData():

    NumberOfFrames = 1007

    height = 1080 # 720
    width = 1920 # 1280
    totalSize = int(height * width)
    nextTotalSize = int(totalSize*1.25) # did when it was doing the yuv stuff dont need it 
    finalTotalSize = int(totalSize*1.5)

    data2 = open("/home/don/Git/LargeDataFilesLinux/ImageData.txt","rb").read() # Change here
    print("Data2 Length: ", len(data2), "Data2 Type: ", type(data2))
    
    data_Values = np.frombuffer(data2, dtype=np.uint8)
    fileHeader_version = int.from_bytes(data2[0:2], "little")
    # print(data_Values[0:5], "Version: ", fileHeader_version)

    data_Values = data_Values[5:] # Remove the File Header Data
    frame_MetaData = []

    for mm in range(0, NumberOfFrames): # Start at 1 because 0 does not exist
        frame_index_start = (mm * finalTotalSize) + 38*mm
        frame_index_end = ((mm) * finalTotalSize) + 38*mm +38
        individual_Meta_Data = data_Values[frame_index_start:frame_index_end]

        frame_MetaData.append(struct.unpack("<HIddffff", individual_Meta_Data))
    
    # print("Done Get All Frame MetaData")

    return frame_MetaData, height, width    

if __name__ == "__main__":
    
    # Get all metadata needed
    allMetaData, imageHeight, imageWidth = getAllframe_metaData()
    print("Finished Processing Metadata")

    # Get all the reshaped frame data needed
    for frame in range(759,1006):
        reshape_data = getframe_data(frame)
    print("Finished Processing Frame Data")


    # half_height = int(imageHeight/ 2)
    # half_width = int(imageWidth / 2)

    # towerHeight, height_imageframe, bounding_box_height, select_frame_height = find_tower_height(half_height, half_width)
    
    # indx_height = height_imageframe.index(select_frame_height)
    # select_bounding_box_height = bounding_box_height[indx_height]
    # select_towerHeight = towerHeight[indx_height]

    # tower_center_x, tower_center_y, center_imageframe, select_frame_center, bounding_box_center = find_tower_center(half_height, half_width)
    
    # selected_frame_metaData = allMetaData[select_frame_center - 1]

    # tower_lat = selected_frame_metaData[2]
    # tower_lon = selected_frame_metaData[3]
    # drone_height = selected_frame_metaData[4]
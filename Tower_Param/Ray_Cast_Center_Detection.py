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
        mask_frame_URL = "/home/don/Git/LargeDataFilesLinux/Images_Frames_First Half/image_frame_" + frame_number + ".jpg"
    else:
        frame_number = str(frameNumber)
        mask_frame_URL = "/home/don/Git/LargeDataFilesLinux/Images_Frames_Second Half/image_frame_" + frame_number + ".jpg"
    
    # print(mask_frame_URL)
    mask_image = Image.open(mask_frame_URL) 

    pixeldata = mask_image.getdata()
    #mask_image.show()


    completepixeldata.append(pixeldata)
    #reshape_data = np.reshape(pixeldata, (mask_image.height, mask_image.width))
    
    # print("Done Get Frame Data")
    return mask_image

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
    #print(allMetaData[759-2:800])
    print("Finished Processing Metadata")

    completepixeldata = []
    # Get all the reshaped frame data needed
    for frame in range(759,1006):
        getframe_data(frame)
    #print(completepixeldata)
    print("Finished Processing Frame Data")

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
    
    print("Done Get All Frame MetaData")

    return frame_MetaData, height, width    

def find_tower_height(half_height, half_width):
   
    tower_Height = []
    diff_y_pix = imageHeight
    bbox = []
    imageFrame = []
    select_frame = 0

    for frame in range(1, 758):
        frameNumber = frame
        selectedFrame = frameNumber - 1
        
        selected_frame_metaData = allMetaData[selectedFrame]

        pixelData = getframe_data(frameNumber)      

        sumdata_x = np.sum(pixelData/255, axis=0)
        min_x = np.nonzero(sumdata_x)[0][0]
        max_x = np.nonzero(sumdata_x)[0][-1]

        # Find the column with most white pixels
        maximum_index = np.argmax(sumdata_x) # The column with the highest pixel count
        maximum_value = sumdata_x[maximum_index] # The number of pixels in column, height

        sumdata_y = np.sum(pixelData/255, axis=1)
        min_y = np.nonzero(sumdata_y)[0][0]
        max_y = np.nonzero(sumdata_y)[0][-1]

        # print(maximum_value, max_y - min_y)

        diff_x_pix_temp = maximum_index - half_width
        diff_y_pix_temp = maximum_value - half_height
        
        if abs(diff_y_pix_temp) < 50:

            # print("frameNum: ", selected_frame_metaData[0], "frameLen: ", selected_frame_metaData[1], "cameraLat: ", selected_frame_metaData[2], "cameraLon: ", 
            # selected_frame_metaData[3], "cameraheight: ", selected_frame_metaData[4], "cameraYaw: ", selected_frame_metaData[5], "cameraPitch: ",
            # selected_frame_metaData[6], "cameraRoll: ", selected_frame_metaData[7])

        #    print(diff_x_pix_temp, diff_y_pix_temp)
           
           bbox.append([min_x, max_x, min_y, max_y])
           imageFrame.append(frame)
           tower_Height.append(selected_frame_metaData[4])

           if (diff_y_pix_temp < diff_y_pix):
                diff_y_pix = diff_y_pix_temp
                select_frame = frame
        
    print("Get Tower Height Done")
    return tower_Height, imageFrame, select_frame, bbox

def find_tower_center(half_height, half_width):
    center_x_diff = []
    center_y_diff = []
    center_diff = 1000
    center_frame_num = []   

    select_frame = 0
    bbox = []

    for frame in range(800, 900):
        frameNumber = frame

        pixelData = getframe_data(frameNumber)

        imageData2 = np.zeros((imageHeight, imageWidth,1), dtype="uint8")
        imageData2 = pixelData.astype("uint8")
        new_image2 = Image.fromarray(imageData2) 

        edges_image = new_image2.filter(ImageFilter.FIND_EDGES)

        edge_data = np.asarray(edges_image)

        sumdata_x = np.sum(edge_data, axis=0)
        min_x = np.nonzero(sumdata_x)[0][0]
        max_x = np.nonzero(sumdata_x)[0][-1]

        sumdata_y = np.sum(edge_data, axis=1)
        min_y = np.nonzero(sumdata_y)[0][0]
        max_y = np.nonzero(sumdata_y)[0][-1]
        
        center_x_temp = min_x + (max_x - min_x) / 2
        center_y_temp = min_y + (max_y - min_y) / 2

        x_diff = (center_x_temp - half_width)
        y_diff = (center_y_temp - half_height)
        diff_len = math.sqrt(x_diff**2 + y_diff**2)

        print(center_x_temp, center_y_temp, diff_len)

        if diff_len < 200:

            center_x_diff.append(x_diff)
            center_y_diff.append(y_diff)
            center_frame_num.append(frame)
            bbox.append([min_x, max_x, min_y, max_y])

            diff_from_center = math.sqrt(x_diff**2 + y_diff**2)

            if diff_from_center < center_diff:
                select_frame = frame

                center_diff = diff_from_center

        
    return center_x_diff, center_y_diff, center_frame_num, select_frame, bbox




if __name__ == "__main__":
    
    # Set up the needed variables and functions needed from adam script
    allMetaData, imageHeight, imageWidth = getAllframe_metaData()
    half_height = int(imageHeight/ 2)
    half_width = int(imageWidth / 2)

    towerHeight, height_imageframe, bounding_box_height, select_frame_height = find_tower_height(half_height, half_width)
    
    indx_height = height_imageframe.index(select_frame_height)
    select_bounding_box_height = bounding_box_height[indx_height]
    select_towerHeight = towerHeight[indx_height]

    tower_center_x, tower_center_y, center_imageframe, select_frame_center, bounding_box_center = find_tower_center(half_height, half_width)
    
    selected_frame_metaData = allMetaData[select_frame_center - 1]

    tower_lat = selected_frame_metaData[2]
    tower_lon = selected_frame_metaData[3]
    drone_height = selected_frame_metaData[4]

    getframe_data()
    print("Inside Ray Cast Detection main")
from re import T
from PIL import Image, ImageDraw, ImageFilter
from PIL.ExifTags import TAGS, GPSTAGS
from matplotlib.transforms import Bbox
import numpy as np
import math 
import struct
import time
import matplotlib.pyplot as plt

def get_decimal_from_dms(dms, ref):

    degrees = dms[0]
    minutes = dms[1] / 60.0
    seconds = dms[2] / 3600.0

    if ref in ['S', 'W']:
        degrees = -degrees
        minutes = -minutes
        seconds = -seconds

    return round(degrees + minutes + seconds, 9)

def get_coordinates(geotags):
    lat = get_decimal_from_dms(geotags['GPSLatitude'], geotags['GPSLatitudeRef'])
    lon = get_decimal_from_dms(geotags['GPSLongitude'], geotags['GPSLongitudeRef'])

    return (lat, lon)

def get_new_Lat_Lon(x, y, dronePos, tower):

    x = x / 1000
    y = y / 1000
    newLat = dronePos[0] + x / 110.574
    newLon = dronePos[1] + y / (111.32*math.cos(newLat * math.pi / 180))

    latError = tower[0] - newLat
    lonError = tower[1] - newLon

    return (newLat, newLon, latError, lonError)

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

def find_image_pixel_ratio(half_imageHeight, half_imageWidth, height):
    # FOV_x = 65 * math.pi / 180
    # FOV_y = 73 * math.pi / 180
    FOV_x = 2 * math.atan((640/2)/527) # Width
    FOV_y = 2 * math.atan((480/2)/527)  # Height

    half_fov_x = FOV_x / 2
    half_fov_y = FOV_y / 2

    pix_x = (height * math.tan(half_fov_x)) / half_imageWidth
    pix_y = (height * math.tan(half_fov_y)) / half_imageHeight

    print("Done Get Pix Ratio")
    return pix_x, pix_y


if __name__ == "__main__":
    actualTower_Lat = get_decimal_from_dms([27, 31, 48.731846], "S")
    actualTower_Lon = get_decimal_from_dms([152, 49, 56.592026], "E")
    print("actualTower_Lat: ", actualTower_Lat, "actualTower_Lon: ", actualTower_Lon)

    allMetaData, imageHeight, imageWidth  = getAllframe_metaData()
    
    half_height = int(imageHeight/ 2)
    half_width = int(imageWidth / 2)
    print("half_height: ", half_height, "half_width: ", half_width)

    # towerHeight, height_imageframe, bounding_box_height, select_frame_height = find_tower_height(half_height, half_width)
    towerHeight = [-34.86970520019531, -34.87581253051758, -34.86766052246094, -36.25898361206055]
    height_imageframe = [200, 201, 202, 754]
    bounding_box_height = [[590, 1185, 491, 1079], [591, 1185, 497, 1079], [591, 1186, 500, 1079], [621, 1350, 561, 1079]]
    select_frame_height = 754

    indx_height = height_imageframe.index(select_frame_height)
    select_bounding_box_height = bounding_box_height[indx_height]
    select_towerHeight = towerHeight[indx_height]

    # tower_center_x, tower_center_y, center_imageframe, select_frame_center, bounding_box_center = find_tower_center(half_height, half_width)
    tower_center_x = [-155.5, -152.5, -150.0, -148.5, -150.0, -157.0, -163.0, -158.5, -158.0, -155.5]  
    tower_center_y = [125.0, 114.0, 55.5, 28.0, 1.0, -30.5, -50.0, -71.0, -92.5, -112.5]
    center_imageframe = [871, 872, 873, 874, 875, 876, 877, 878, 879, 880]
    bounding_box_center = [[570, 1039, 411, 919], [574, 1041, 429, 879], [574, 1046, 335, 856], [573, 1050, 306, 830], [572, 1048, 280, 802], 
          [566, 1040, 250, 769], [563, 1031, 228, 752], [558, 1045, 210, 728], [560, 1044, 189, 706], [559, 1050, 167, 688]]
    select_frame_center = 875

    selected_frame_metaData = allMetaData[select_frame_center - 1]

    tower_lat = selected_frame_metaData[2]
    tower_lon = selected_frame_metaData[3]
    drone_height = selected_frame_metaData[4]
    
    # print(bounding_box_center)
    print("Tower Center X: ", tower_center_x, "Tower Center Y: ", tower_center_y, "Image Frame: ", center_imageframe, "Select Frame: ", select_frame_center)   

    indx_center = center_imageframe.index(select_frame_center)
    select_bounding_box_center = bounding_box_center[indx_center]

    diff_towerHeight = abs(drone_height) - abs(select_towerHeight)
    print("diff_towerHeight: ", diff_towerHeight)

    ratio_pix_x, ratio_pix_y = find_image_pixel_ratio(half_height, half_width, diff_towerHeight)
    print("ratio_pix_x: ", ratio_pix_x, "ratio_pix_y: ", ratio_pix_y)
    
    pix_length_x = select_bounding_box_center[1] - select_bounding_box_center[0]
    pix_length_y = select_bounding_box_center[3] - select_bounding_box_center[2]
    
    length_x = pix_length_x * ratio_pix_x
    length_y = pix_length_y * ratio_pix_y
    diameter = math.sqrt(length_x**2 + length_y**2)

    pixelData_height = getframe_data(select_frame_height)
    pixel_image_height = Image.fromarray(np.asarray(pixelData_height.astype("uint8"))) 
    draw = ImageDraw.Draw(pixel_image_height)
    rad = 15
    draw.ellipse((half_width - rad, half_height - rad, half_width + rad, half_height + rad), fill=(125))
    draw.rectangle((select_bounding_box_height[0], select_bounding_box_height[2], select_bounding_box_height[1], select_bounding_box_height[3]), outline =(255))
    # plt.imshow(pixel_image_height, cmap="gray")
    # plt.show()

    pixelData_center = getframe_data(select_frame_center)
    pixel_image_center = Image.fromarray(np.asarray(pixelData_center.astype("uint8"))) 
    draw = ImageDraw.Draw(pixel_image_center)
    rad = 15
    draw.ellipse(((tower_center_x[indx_center] + half_width - rad), (tower_center_y[indx_center] + half_height - rad), (tower_center_x[indx_center] + half_width + rad), (tower_center_y[indx_center] + half_height + rad)), fill=(0))
    draw.ellipse((half_width - rad, half_height - rad, half_width + rad, half_height + rad), fill=(125))
    draw.rectangle((select_bounding_box_center[0], select_bounding_box_center[2], select_bounding_box_center[1], select_bounding_box_center[3]), outline =(255))
    # plt.imshow(pixel_image_center, cmap="gray")
    # plt.show()
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(pixel_image_height, cmap="gray")
    ax.set_title("Height: \n" + str(round(abs(select_towerHeight), 4)))
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(pixel_image_center, cmap="gray")
    ax.set_title("Center: \n[" + str(round(tower_lat, 8)) + " , " + str(round(tower_lon, 8)) + "]")

    plt.show()

    print("Tower Height: ", select_towerHeight, "Image Frame: ", select_frame_height)
    print("tower_lat: ", tower_lat, "tower_lon: ", tower_lon, "Drone Height: ", drone_height)
    print("length_x [m]: ", length_x, "length_y [m]: ", length_y, " Diameter [m]: ", diameter, "Radius [m]: ", diameter/2)

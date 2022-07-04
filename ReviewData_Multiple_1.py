from PIL import Image
import numpy as np
import cv2 as cv

NumberOfFrames = 247
CurrentFrameNumber = 0

def render_one_frame(CurrentFrameNumber):
    height = 720 # 720
    width = 1280 # 1280
    totalSize = int(height * width)
    nextTotalSize = int(totalSize*1.25)
    finalTotalSize = int(totalSize*1.5)
    print("Frame TotalSize: ", totalSize, "Frame TotalSize*1.25: ", nextTotalSize, "Frame FinalDataSize: ", finalTotalSize)

    # data2 = open("/Users/don/GitHub/ImageProcessing/ImageDataMultipleFrames.txt","rb").read() #.splitlines() # , encoding= "utf-8" 
    # data2 = open("/Users/don/GitHub/ImageProcessing/ImageDatav2.txt","rb").read() #.splitlines() # , encoding= "utf-8" 
    #data2 = open("/Users/don/GitHub/ImageProcessing/ImageDatav3.txt","rb").read() #.splitlines() # , encoding= "utf-8" # Line 24 is 36 , Line 26 same, Line 33 is the frame you want to look at 
    data2 = open("/Users/don/GitHub/ImageProcessing/ImageDatav4.txt","rb").read() #.splitlines() # , encoding= "utf-8" # Line 24 is 36 , Line 26 same, Line 33 is the frame you want to look at
    print("Data2 Length: ", len(data2), "Data2 Type: ", type(data2))

    data_Values = np.frombuffer(data2, dtype=np.uint8)
    data_Values = data_Values.astype(np.float64)
    data_Values = data_Values[5:]
    print("data_Values Length: ", len(data_Values), "data_Values Type: ", type(data_Values))
    print(data_Values)



    individualData = np.zeros((248,finalTotalSize), dtype=np.float64) # Change the number in X -> (X,finalTotalSize)

    for mm in range(248): # Number of frames to iterate through. change the number in X -> range(X)
        frame_index_start = (mm * finalTotalSize) + 38*mm + 38
        frame_index_end = ((mm+1) * finalTotalSize) + 38*mm +38
        individualData[mm, 0:finalTotalSize] = data_Values[frame_index_start:frame_index_end]
    
    data = individualData[CurrentFrameNumber, 0:finalTotalSize] # Which frame you want to use change the number in X -> [X,0:finalTotalSize]
    print("Data Length: ", len(data), "Data Type: ", data.dtype, "Data Shape: ", data.shape)

    YColour = np.reshape(data[0:totalSize],(height, width))
    UColour = np.reshape(data[totalSize:finalTotalSize:2], (int(height/2), int(width/2)))
    VColour = np.reshape(data[(totalSize+1):finalTotalSize:2], (int(height/2), int(width/2)))
    
    print("Y", YColour.shape, np.max(YColour), np.min(YColour))
    print("U", UColour.shape, np.max(UColour), np.min(UColour))
    print("V", VColour.shape, np.max(VColour), np.min(VColour))

    repeatUColour_1 = np.repeat(UColour, 2, axis=1) # Repeat each value twice along the rows
    repeatVColour_1 = np.repeat(VColour, 2, axis=1) # Repeat each value twice along the rows

    repeatUColour_2 = np.repeat(repeatUColour_1, 2, axis=0) # Repeat each value twice along the cols
    repeatVColour_2 = np.repeat(repeatVColour_1, 2, axis=0) # Repeat each value twice along the cols

    UFull = np.resize(repeatUColour_2, (height, width))
    VFull = np.resize(repeatVColour_2, (height, width))

    print("UFull", UFull.shape)
    print("VFull", VFull.shape)

    YImageColour = np.clip(YColour,0,255)
    UImageColour = np.clip(UFull,0,255)
    VImageColour = np.clip(VFull,0,255)

    R = np.clip(((YImageColour) + (1.402 * (VImageColour-128))), 0, 255)
    G = np.clip(((YImageColour) - (0.344 * (UImageColour-128)) - (0.714 * (VImageColour-128))), 0, 255)
    B = np.clip(((YImageColour) + (1.772 * (UImageColour-128))), 0, 255)

    print("R", R.shape, np.max(R), np.min(R))
    print("G", G.shape, np.max(G), np.min(G))
    print("B", B.shape, np.max(B), np.min(B))


    imageData = np.zeros((height, width, 3), dtype="uint8")

    imageData[:,:,0] = YImageColour.astype("uint8")
    imageData[:,:,1] = UImageColour.astype("uint8")
    imageData[:,:,2] = VImageColour.astype("uint8")

    # new_image_Y = Image.fromarray(imageData[:,:,0]) #, mode="YCbCr"
    # new_image_Y.show()

    # new_image_U = Image.fromarray(imageData[:,:,1]) #, mode="YCbCr"
    # new_image_U.show()

    # new_image_V = Image.fromarray(imageData[:,:,2]) #, mode="YCbCr"
    # new_image_V.show()

    # new_image1 = Image.fromarray(imageData, mode="YCbCr") #, mode="YCbCr"
    # new_image1.show()

    imageData2 = np.zeros((height, width, 3), dtype="uint8")

    imageData2[:,:,0] = R.astype("uint8")
    imageData2[:,:,1] = G.astype("uint8")
    imageData2[:,:,2] = B.astype("uint8")

    new_image2 = Image.fromarray(imageData2) #, mode="RGB"
    #new_image2.show()
    gif.append(new_image2)
    

if __name__ == "__main__":
    images = []
    gif = []
    for x in range(NumberOfFrames):
        render_one_frame(CurrentFrameNumber)
        CurrentFrameNumber = CurrentFrameNumber + 1
    gif[0].save('temp_result.gif', save_all=True,optimize=False, append_images=gif[1:], loop=0)


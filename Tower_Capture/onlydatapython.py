from PIL import Image
import numpy as np
data = open("/Users/don/Desktop/ImageProcessing/ImageData_0.txt", "rb").read().splitlines()
print("Data Length: ", len(data))
height = 720 # 640 , 720
width = 1280 # 720, 1280
totalSize = height * width 
print("Total Size: ", totalSize)
count = 0
YColour = []
UColour = []
VColour = []
redColour = []
greenColour = []
blueColour = []
        
for jj in range(height):
    for ii in range(width):
        # Y = int(data[ii * width + jj])
        # U = int(data[int((ii / 2) * (width / 2) + (jj / 2) + width)])
        # V = int(data[int((ii / 2) * (width / 2) + (jj / 2) + totalSize + (totalSize / 4))])
        Y = int(data[jj * width + ii])
        U = int(data[int((jj / 2) * (width / 2) + (ii / 2) + totalSize)])
        #V = int(data[int((jj/2) * (width/2) + (ii/2) + (totalSize + totalSize/4))])
        #print(((jj / 2) * (width / 2) + (ii / 2) + totalSize), ((jj/2) * (width/2) + (ii/2) + (totalSize + totalSize/4)))
        YColour.append(Y)
        UColour.append(U)
        # VColour.append(V)
                 
        # R = (Y) + (1.402 * (V - 128))
        # G = (Y) - (0.344 * (U - 128)) - (0.714 * (V - 128))
        B = (Y) + (1.772 * (U - 128))
        # redColour.append(R)
        # greenColour.append(G)
        blueColour.append(B)
        
        count = count + 1  
        # print("R: \(R), G: \(G), B: \(B), Y: \(Y), U: \(U), V: \(V),")
    
print("Current Count: ", count)
print(len(YColour))
print(len(UColour))
print(len(VColour))
print(len(redColour))
print(len(greenColour))
print(len(blueColour))
# YData = []
# for kk in range(totalSize):
#     YData.append(int(data[kk]))
# UData = []
# print(UColour)
imageData = np.zeros((height, width, 3), dtype=np.uint8)
imageData[:,:,0] = (np.resize(YColour, (height, width)))
imageData[:,:,1] = (np.resize(UColour, (height, width)))
imageData[:,:,2] = np.resize(VColour, (height, width))
new_image1 = Image.fromarray(imageData)
new_image1.show()
imageData2 = np.zeros((height, width, 3), dtype=np.uint8)
imageData2[:,:,0] = np.resize(redColour, (height, width))
imageData2[:,:,1] = np.resize(greenColour, (height, width))
imageData2[:,:,2] = np.resize(blueColour, (height, width))
new_image2 = Image.fromarray(imageData2)
new_image2.show()

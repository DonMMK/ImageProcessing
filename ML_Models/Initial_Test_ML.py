import coremltools as ct
import copy
import random
import time

from PIL import Image, ImageColor, ImageDraw
import numpy as np
import pathlib
import cv2

## VISUALIZATION     
def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[:, 0]), int(x[:, 1])), (int(x[:, 2]), int(x[:, 3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    cv2.imshow("img", img)
    # cv2.imwrite("Prediction_Image_"+str(selectframe)+".jpg", img)
    cv2.waitKey(0)

def resize_image(original_image, resize_target=640):
    h, w = original_image.size[:2]
    image_max = max(h, w)
    scale = resize_target / image_max
    image_resized = original_image.resize(
        size=(round(h * scale), round(w * scale)),
        resample=Image.Resampling.LANCZOS)
    ## padding image to keep aspect-ratio
    return padding_img(image_resized, (0,0,0))

def padding_img(image_resized, background_color):
    width, height = image_resized.size
    if width == height:
        return image_resized
    elif width > height:
        result = Image.new(image_resized.mode, (width, width), background_color)
        result.paste(image_resized, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(image_resized.mode, (height, height), background_color)
        result.paste(image_resized, ((height - width) // 2, 0))
        return result

def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clip(0, img_shape[1])  # x1
    boxes[:, 1].clip(0, img_shape[0])  # y1
    boxes[:, 2].clip(0, img_shape[1])  # x2
    boxes[:, 3].clip(0, img_shape[0])  # y2

def scale_coords(resize_image, coords, original_image, ratio_pad=None):
    # Rescale coords (xyxy) from resize_image to original_image
    if ratio_pad is None:  # calculate from original_image
        gain = min(resize_image[0] / original_image[0], resize_image[1] / original_image[1])  # gain  = old / new
        pad = (resize_image[1] - original_image[1] * gain) / 2, (resize_image[0] - original_image[0] * gain) / 2  # wh padding    
    coords[:,[0, 2]] -= pad[0]  # x padding
    coords[:,[1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, original_image)
    return coords

def convert_xywh_xyxy(coords_yxwh):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    coords_xyxy = np.empty_like(coords_yxwh)
    coords_xyxy[:, 0] = coords_yxwh[:, 0] - coords_yxwh[:, 2] / 2  
    coords_xyxy[:, 1] = coords_yxwh[:, 1] - coords_yxwh[:, 3] / 2
    coords_xyxy[:, 2] = coords_yxwh[:, 0] + coords_yxwh[:, 2] / 2
    coords_xyxy[:, 3] = coords_yxwh[:, 1] + coords_yxwh[:, 3] / 2
    return coords_xyxy


selectframe = 538
selectImage = "image_frame_" + str(selectframe) + ".jpg"


ts = time.time()
original_image = Image.open(pathlib.Path(selectImage))
resize_target=640
image_resize = resize_image(original_image, resize_target)


cml_model = ct.models.MLModel('yolov7-640.mlmodel')
preds = cml_model.predict(data={'image': image_resize, 'iouThreshold':0.3, 'confidenceThreshold':0.3})
print("predictions {}".format(preds))

# the coordinates are normalize by the image_resize = 640
coords_yxwh = preds['coordinates']*resize_target 

coords_xyxy = convert_xywh_xyxy(coords_yxwh)

coords_xyxy[:, :4] = scale_coords(np.array(image_resize).shape, coords_xyxy[:, :4], np.array(original_image).shape).round()
te = time.time()
print("coords_xyxy {}".format(coords_xyxy))
print("elapse time {}".format(te-ts))

plot_one_box(coords_xyxy, np.array(original_image), label=f'{preds["confidence"][0]}_tower',  line_thickness=3)
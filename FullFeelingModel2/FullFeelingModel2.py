import json
import pickle
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from collections import OrderedDict
import colour
from colour_checker_detection import (
    EXAMPLES_RESOURCES_DIRECTORY,
    colour_checkers_coordinates_segmentation,
    detect_colour_checkers_segmentation)
from colour_checker_detection.detection.segmentation import (
    adjust_image)
colour.plotting.colour_style()

# The function of changing the image resolution

def viewResize(image,width,height):
    dsize = (width, height)
    output = cv2.resize(image, dsize, interpolation = cv2.INTER_AREA)
    return output

# The function of finding the target area of the image and finding colors for black-white correction

def color_detection_and_cropping_zone(img):
    default_height = img.shape[0]
    default_width = img.shape[1]
    new_height = default_height
    new_width = default_width
    temp_size = int(70*default_height/6000)
    image = viewResize(img,new_width,new_height)
    blur_coef = int(21*new_height/6000)
    if blur_coef&1 == 0:
        blur_coef+=1
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (blur_coef, blur_coef), 0)
    edged = cv2.Canny(gray, 15, 75)
    coef_w = int(default_width/new_width)
    coef_h = int(default_height/new_height)
    param_1 = int(temp_size/coef_h)+1
    param_2 = int(temp_size/coef_w)+1


    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (param_1, param_2))
    closed  = cv2.dilate(edged, kernel, iterations = 1)
    result = cv2.bitwise_and(image, image, mask=closed)
    points = np.zeros((4,2)).astype(int)



    array = np.array(closed)
    flag = 5
    for j in range(0,default_height-1,1):
        if flag!=0 and closed[j,int(default_width/2)]!=closed[j+1,int(default_width/2)]:
            points[0,0] = points[0,1]
            points[0,1] = j
            flag-=1
    points[0,1] = int((points[0,0]+points[0,1])/2)
    test = np.array(image)
    value_1 = test[points[0,0],int(test.shape[1]/2)]

    flag = 3
    for j in range(default_height-1,1,-1):
        if flag!=0 and closed[j,int(default_width/2)]!=closed[j-1,int(default_width/2)]:
            points[1,1] = points[1,0]
            points[1,0] = j
            flag-=1
    points[1,0] = int((points[1,0]+points[1,1])/2)
    test = np.array(image)
    value_1 = test[points[1,0],int(test.shape[1]/2)]

    flag = 5
    for j in range(0,default_width-1,1):
        if flag!=0 and closed[int(default_height/2),j]!=closed[int(default_height/2),j+1]:
            points[2,0] = points[2,1]
            points[2,1] = j
            flag-=1
    points[2,1] = int((points[2,0]+points[2,1])/2)
    test = np.array(image)
    value_1 = test[int(test.shape[0]/2),points[2,0]]

    flag = 3
    for j in range(default_width-1,1,-1):
        if flag!=0 and closed[int(default_height/2),j]!=closed[int(default_height/2),j-1]:
            points[3,1] = points[3,0]
            points[3,0] = j
            flag-=1
    points[3,0] = int((points[3,0]+points[3,1])/2)
    test = np.array(image)
    value_1 = test[int(test.shape[0]/2),points[3,1]]


    q = points[0,1]-points[0,0]
    w = points[1,1]-points[1,0]
    e = points[2,1]-points[2,0]
    r = points[3,1]-points[3,0]
    borders_points = np.array([[points[0,0],points[2,0]+w*3],[points[0,1],points[3,1]-r*3], #top-rect
                        [points[1,0],points[2,0]+w*3],[points[1,1],points[3,1]-r*3], #bottom-rect
                        [points[0,0]+q*3,points[2,0]],[points[1,1]-e*3,points[2,1]],#left-rect
                        [points[0,0]+q*3,points[3,0]],[points[1,1]-e*3,points[3,1]],#right-rect
    ])
    array = np.array(image)
    border_colors = np.zeros((4,3))
    for ii in range(0,8,2):
        top_left = np.array([borders_points[ii,1],borders_points[ii,0]])
        low_right = np.array([borders_points[ii+1,1],borders_points[ii+1,0]])
        crop = image[top_left[1]:low_right[1],top_left[0]:low_right[0]]
        border_colors[int(ii/2)]=np.array([np.median(crop[:,:,0]),np.median(crop[:,:,1]),np.median(crop[:,:,2])]).astype(int)
    purpose_zone = image[int(points[0,1]+q*2):int(points[1,0]-w*2.0),int(points[2,1]+e*1.5):int(points[3,0]-r*2.0)]
    return purpose_zone,border_colors

# The function of bringing the image colors to the range [0 255] after color correction

def normalized(image):
    corrected_image = np.array(image).astype("int")
    min_value = np.zeros((image.shape[0],image.shape[1]))+0
    max_value = np.zeros((image.shape[0],image.shape[1]))+255
    r = np.zeros((image.shape[0],image.shape[1],2))
    g = np.zeros((image.shape[0],image.shape[1],2))
    b = np.zeros((image.shape[0],image.shape[1],2))

    r[:,:,0] = max_value
    r[:,:,1] = corrected_image[:,:,0]
    min_r = np.amin(r, axis = 2)
    r[:,:,0] = min_value
    r[:,:,1] = min_r
    max_r = np.amax(r, axis = 2)
    g[:,:,0] = max_value
    g[:,:,1] = corrected_image[:,:,1]
    min_g = np.amin(g, axis = 2)
    g[:,:,0] = min_value
    g[:,:,1] = min_g
    max_g = np.amax(g, axis = 2)
    b[:,:,0] = max_value
    b[:,:,1] = corrected_image[:,:,2]
    min_b = np.amin(b, axis = 2)
    b[:,:,0] = min_value
    b[:,:,1] = min_b
    max_b = np.amax(b, axis = 2)
    corrected_image[:,:,0] = max_r
    corrected_image[:,:,1] = max_g
    corrected_image[:,:,2] = max_b

    corrected_image = np.array(corrected_image).astype(np.uint8)
    return corrected_image

# Calculating the color correction matrix

REFERENCE_IMAGE = ['C:\\normal.png']
ORIGIN_IMAGE = ['C:\\checker.png']

COLOUR_CHECKER_IMAGE_PATHS = ORIGIN_IMAGE
COLOUR_CHECKER_REFERENCE_IMAGE_PATHS = REFERENCE_IMAGE
COLOUR_CHECKER_IMAGES = [
    colour.cctf_decoding(colour.io.read_image(path))
    for path in COLOUR_CHECKER_IMAGE_PATHS
]
COLOUR_CHECKER_REFERENCE_IMAGES = [
    colour.cctf_decoding(colour.io.read_image(path))
    for path in COLOUR_CHECKER_REFERENCE_IMAGE_PATHS
]

SWATCHES = []
for image in COLOUR_CHECKER_IMAGES:
    for swatches, colour_checker, masks in detect_colour_checkers_segmentation(image, additional_data=True):
        SWATCHES.append(swatches)

print('***********\n***Origin***\n***********')
print(SWATCHES)

REFERENCE_SWATCHES =[]
for image in COLOUR_CHECKER_REFERENCE_IMAGES:
    for swatches, colour_checker, masks in detect_colour_checkers_segmentation(
        image, additional_data=True):
        REFERENCE_SWATCHES.append(swatches)

    
print('***********\n*Reference*\n***********') 
REFERENCE_SWATCHES = np.array(REFERENCE_SWATCHES)[0,:,:]
print(REFERENCE_SWATCHES)

D65 = colour.ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']
REFERENCE_COLOUR_CHECKER = colour.COLOURCHECKERS['ColorChecker 2005']


for i, swatches in enumerate(SWATCHES):
    swatches_xyY = colour.XYZ_to_xyY(colour.RGB_to_XYZ(
        swatches, D65, D65, colour.RGB_COLOURSPACES['sRGB'].RGB_to_XYZ_matrix))


    colour_checker = colour.characterisation.ColourChecker(
        os.path.basename(COLOUR_CHECKER_IMAGE_PATHS[i]),
        OrderedDict(zip(REFERENCE_COLOUR_CHECKER.data.keys(), swatches_xyY)),
        D65)
    
    swatches_f = colour.colour_correction(swatches, swatches, REFERENCE_SWATCHES)
    swatches_f_xyY = colour.XYZ_to_xyY(colour.RGB_to_XYZ(
        swatches_f, D65, D65, colour.RGB_COLOURSPACES['sRGB'].RGB_to_XYZ_matrix))
    colour_checker = colour.characterisation.ColourChecker(
        '{0} - CC'.format(os.path.basename(COLOUR_CHECKER_IMAGE_PATHS[i])),
        OrderedDict(zip(REFERENCE_COLOUR_CHECKER.data.keys(), swatches_f_xyY)),
        D65)

directory ='C:\\my_volume_input\\'
files = os.listdir(directory)
files = filter(lambda x: x.endswith('.json'), files)
for file in files:
    js_file = file


with open(directory+str(js_file), "r") as read_file:
    data = json.load(read_file)

accept = str(data['input']['session_config']['accept'])[2:-2]
reject = str(data['input']['session_config']['reject'])[2:-2]


with open("C:\\meta_file.json", "r") as read_file:
    meta = json.load(read_file)


yellow_set = set(meta["model_parameters"][accept])
black_set = set(meta["model_parameters"][reject])


files = os.listdir(directory)
js_file = [js_file]
files=list(set(files)-set(list(js_file)))

filename = 'GM_model.sav'
loaded_model = pickle.load(open("C:\\GM_model.sav", 'rb'))
class_count = 5
spam = list(range(0, class_count))

new_height = 3024
new_width = 4032


sum_wheat_part = 0
sum_wheat_area = 0
sum_raps_part = 0
sum_raps_area = 0


data["output"] = []
data_test = {}
print(files)

# Analysis and formation of the output file

for file in files:
    image = cv2.imread(directory+file)
    output,border_colors = color_detection_and_cropping_zone(image)
    arr_image = np.array(output).astype(float)/255
    img = colour.cctf_encoding(
        colour.colour_correction(
            arr_image, swatches, REFERENCE_SWATCHES))*255
    output = normalized(img)

    arr = np.array(output)
    r = arr[:,:,0].flatten()
    g = arr[:,:,1].flatten()
    b = arr[:,:,2].flatten()
    
    df = pd.DataFrame()
    df["red"]= r
    df["green"]= g
    df["blue"]= b

    Score = loaded_model.predict(df)
    Score_matrix = Score.reshape(arr.shape[0],arr.shape[1])

    score = pd.DataFrame(Score)
    score_1 = pd.DataFrame({0:spam})
    score = score.append(score_1)
    
    
    sum = score.value_counts().sort_index()
    for i in range(class_count):
        sum[i]-=1
    sum = sum.where(sum>=0).dropna().astype(int)


    wheat_part = sum[yellow_set].sum()/(sum[yellow_set].sum()+sum[black_set].sum())
    wheat_area = sum[yellow_set].sum()
    raps_part = sum[black_set].sum()/(sum[yellow_set].sum()+sum[black_set].sum())
    raps_area = sum[black_set].sum()
    
    sum_wheat_part += wheat_part
    sum_wheat_area += wheat_area
    sum_raps_part += raps_part
    sum_raps_area += raps_area

    data_test[file] = []
    data_test[file].append({"part":[{accept:float(int(wheat_part*10000)/10000)},{reject:float(int(raps_part*10000)/10000)}],
                                  "area":[{accept:int(wheat_area)},{reject:int(raps_area)}]
                                  })

data["output"] ={"analysis_results":{"accept": [ { "accept": float(int((sum_wheat_part/(sum_wheat_part + sum_raps_part))*10000)/10000) } ],
                         "reject": [ { "reject": float(int((sum_raps_part/(sum_wheat_part + sum_raps_part))*10000)/10000) } ],
                         "part_total":[{accept: float(int((sum_wheat_part/(sum_wheat_part + sum_raps_part))*10000)/10000)},{reject: float(int((sum_raps_part/(sum_wheat_part + sum_raps_part))*10000)/10000)}],
                         "area_total":[{accept: int(sum_wheat_area)},{reject: int(sum_raps_area)}],
                         "separately":[data_test]},
                 "image_parameters": {
                         "imWidth": new_width,
                         "imHeight": new_height}}

with open("C:\my_volume_output\data_file.json", "w") as write_file:
    json.dump(data, write_file)
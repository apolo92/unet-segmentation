import unet-model as Unet
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from scipy import ndimage



'''Function to decode encoded mask labels
        Inputs:
            onehot - one hot encoded image matrix (height x width x num_classes)
            colormap - dictionary of color to label id
        Output: Decoded RGB image (height x width x 3)
    '''
def onehot_to_rgb(onehot, colormap):
    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros( onehot.shape[:2]+(3,) )
    for k in colormap.keys():
        output[single_layer==k] = colormap[k]
    return np.uint8(output)

    '''Function to read and resize images 
        Inputs:
            path - directory to read all images
        Output: np array to all images in path pass
    '''
def get_resize_images(path):
    pathImg = list(map(lambda imgName: path + '/' + imgName, os.listdir(path)))
    images = list(map(lambda path: cv2.imread(path), pathImg))
    resizeimg = np.array(list(map(lambda img: cv2.resize(img, (256, 256)), images)))

    return resizeimg

def calculate_dif_image(label_coord, img):
    x = int(label_coord.start * (img.shape[0] / 256))
    y = int(label_coord.stop * (img.shape[1] / 256))
    return x, y

    '''Function to divide image with mask segmentation
        Inputs:
            mask - segmentation mask image
            img - image to divide
        Output: array to differents elements inside img
    '''
def div_img_to_mask(mask, img):
    labels, nlabels = ndimage.label(mask)

    label_arrays = []
    for label_num in range(1, nlabels + 1):
        label_mask = np.where(labels == label_num, 1, 0)
        label_arrays.append(label_mask)

    print('There are {} separate components / objects detected.'.format(nlabels))
    result = []
    for label_ind, label_coords in enumerate(ndimage.find_objects(labels)):
        difImage = tuple(map(lambda label_coord: calculate_dif_image(label_coord, img), label_coords))
        division = img[difImage[0][0]:difImage[0][1], difImage[1][0]:difImage[1][1]]
        # Check if the label size is too small
        if np.product(division.shape) < 10:
            print('Label {} is too small! Setting to 0.'.format(label_ind))
            mask = np.where(labels == label_ind + 1, 0, mask)
        result.append(division)

    return result


#RGB category mask colors
label_codes = [(0, 0, 128), (0, 128, 0), (0, 128, 128), (128, 0, 0), (0, 0, 0)]
#category labels for segmentation
label_names = ['logo', 'info1', 'info2', 'table', 'black']

code2id = {v:k for k,v in enumerate(label_codes)}
id2code = {k:v for k,v in enumerate(label_codes)}

modelo = Unet.unet()
modelo.load_weights('model.hdf5')

predict_img = get_resize_images('data/test/img')
result_predict = modelo.predict(predict_img)

#print predict mask to matplotlib
plt.imshow(onehot_to_rgb(result_predict[0], id2code))

img_div = div_img_to_mask(onehot_to_rgb(result_predict[0], id2code), result_predict[0])

plt.imshow(img_div[0])
plt.imshow(img_div[1])


import unet-model as Unet
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import random


'''Function to one hot encode RGB mask labels
        Inputs:
            rgb_image - image matrix (eg. 256 x 256 x 3 dimension numpy ndarray)
            colormap - dictionary of color to label id
        Output: One hot encoded image of dimensions (height x width x num_classes) where num_classes = len(colormap)
    '''
def rgb_to_onehot(rgb_image, colormap):
    num_classes = len(colormap)
    shape = rgb_image.shape[:2]+(num_classes,)
    encoded_image = np.zeros( shape, dtype=np.int8 )
    for i, cls in enumerate(colormap):
        encoded_image[:,:,i] = np.all(rgb_image.reshape( (-1,3) ) == colormap[i], axis=1).reshape(shape[:2])
    return encoded_image

    '''Function to normalize images
        Inputs:
            images - nparray images
        Output: images normalized in domain 0,1
    '''
def normalization_images(images):
    return images / 256


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

    '''Function to increment data flip images
        Inputs:
            imges - image RGB
            masks - mask images
        Output: same input with new flip images
    '''
def random_turn_img(imges, masks):
    for img, mask in zip(imges, masks):
        if random.uniform(0, 1) > 0.5:
            imges = np.append(imges, [np.flipud(img)], axis=0)
            masks = np.append(masks, [np.flipud(mask)], axis=0)
    return imges, masks


PATHAMEIMG = './data/img'
PATHMASKIMG = './data/mask'

#RGB category mask colors
label_codes = [(0, 0, 128), (0, 128, 0), (0, 128, 128), (128, 0, 0), (0, 0, 0)]
#category labels for segmentation
label_names = ['logo', 'info1', 'info2', 'table', 'black']

code2id = {v:k for k,v in enumerate(label_codes)}
id2code = {k:v for k,v in enumerate(label_codes)}

images = normalization_images(get_resize_images(PATHAMEIMG))
maskImg = get_resize_images(PATHMASKIMG)

images, maskImg = random_turn_img(images, maskImg)

maskImg = np.array(list(map(lambda img: rgb_to_onehot(img, id2code), maskImg)))

modelo = Unet.unet()
modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
modelo.fit(images, maskImg, epochs=20, batch_size=20)

test_image = get_resize_images('./data/test/img')
test_mask = get_resize_images('./data/test/mask')
test_mask = np.array(list(map(lambda img: rgb_to_onehot(img, id2code), test_mask)))
dialec_test, dialec_test_mask = random_turn_img(test_image, test_mask)
print(modelo.evaluate(dialec_test, dialec_test_mask))

modelo.save_weights('model.hdf5')

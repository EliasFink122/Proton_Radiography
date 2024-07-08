# -*- coding: utf-8 -*-
"""
Created on Mon Jul 01 2024

@author: Elias Fink (elias.fink22@imperial.ac.uk)

Cropping and rotating RCF stack images from initial scan
"""
import numpy as np
from PIL import Image

def import_tif(path = 'sh1_st1.tif') -> np.ndarray:
    """
    Import tif image

    Returns:
        numpy array of pixel rgb values with higher contrast
    """
    # import image
    imge = Image.open(path)
    imarray = np.array(imge)

    # increase contrast in the image
    for k, row in enumerate(imarray):
        for j, pixel in enumerate(row):
            if np.mean(pixel) > 180:
                imarray[k][j] = [255, 255, 255]
    return imarray

def slice_image(img: np.ndarray) -> list[np.ndarray]:
    """
    Slices full image roughly into sections of stack images

    Args:
        img: pixels of full input image

    Returns:
        array of sections in order of stack
    """
    # slice horizontally
    slices_x = []
    starts = []
    ends = []
    for k, row in enumerate(img):
        if np.mean(row) <= 254 and np.mean(img[k-1]) > 254:
            starts.append(k)
        elif np.mean(row) > 254 and np.mean(img[k-1]) <= 254:
            ends.append(k)
    for k, start in enumerate(starts):
        if ends[k] - start > 100:
            slices_x.append(img[start:ends[k]])

    # slice vertically
    imgs_sliced = []
    for slice_x in slices_x:
        slice_x = np.transpose(slice_x, (1, 0, 2))
        starts = []
        ends = []
        for k, row in enumerate(slice_x):
            if np.mean(row) <= 254 and np.mean(slice_x[k-1]) > 254:
                starts.append(k)
            elif np.mean(row) > 254 and np.mean(slice_x[k-1]) <= 254:
                ends.append(k)
        for k, start in enumerate(starts):
            if ends[k] - start > 100:
                imgs_sliced.append(slice_x[start:ends[k]].transpose(1, 0, 2))

    # clean up slices horizontally
    for k, img_sliced in enumerate(imgs_sliced):
        start = 0
        end = len(img_sliced) - 1
        for j, row in enumerate(img_sliced):
            if np.mean(row) <= 254 and np.mean(img_sliced[j-1]) > 254:
                start = j
            elif np.mean(row) > 254 and np.mean(img_sliced[j-1]) <= 254 and j != 0:
                end = j
                break
        imgs_sliced[k] = img_sliced[start:end]
    return imgs_sliced

def rotate(imgs: list[np.ndarray]) -> list[np.ndarray]:
    """
    Rotates images by the respective amount to all be in the correct orientation

    Args:
        imgs: list of slices images

    Returns:
        list of correctly orientated images
    """
    imgs_rotated = []
    for img in imgs:
        # find one edge
        edge_x = np.array([])
        edge_y = np.array([])
        for k, row in enumerate(img):
            for j, pixel in enumerate(row):
                if np.mean(pixel) <= 150:
                    edge_x = np.append(edge_x, j)
                    edge_y = np.append(edge_y, k)
                    break

        for k, p_x in enumerate(edge_x):
            if np.abs(p_x - edge_x[k]) >= 5:
                edge_x = edge_x[k:]
                edge_y = edge_y[k:]

        p1 = np.array([edge_x[int(0.1 * len(edge_x))],
                       edge_y[int(0.1 * len(edge_y))]])
        p2 = np.array([edge_x[int(0.9 * len(edge_x))],
                       edge_y[int(0.9 * len(edge_y))]])

        # rotate image to make orientation of edge straight
        theta = np.arctan((p2[0] - p1[0])/(p2[1] - p1[1]))

        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta), np.cos(theta)]])
        new_img = np.full(np.shape(img), 255)
        for k, row in enumerate(img):
            for j, pixel in enumerate(row):
                if np.mean(pixel) != 255:
                    old_xy = np.array([j - len(img[0])/2, k - len(img)/2])
                    new_xy = np.dot(rotation_matrix, old_xy)
                    try:
                        new_j = int(new_xy[0] + len(row)/2)
                        new_i = int(new_xy[1] + len(img)/2)
                        if new_j < 0 or new_i < 0:
                            continue
                        new_img[new_i, new_j] = pixel
                    except IndexError:
                        continue

        # clean up white space on rotated image
        start = 0
        end = len(new_img) - 1
        for j, row in enumerate(new_img):
            if np.mean(row) <= 150 and np.mean(new_img[j-1]) > 150:
                start = j
            elif np.mean(row) > 150 and np.mean(new_img[j-1]) <= 150 and j != 0:
                if j - start > 100:
                    end = j
                break
        new_img = new_img[start:end]

        start = 0
        end = len(new_img.transpose(1, 0, 2)) - 1
        for j, row in enumerate(new_img.transpose(1, 0, 2)):
            if np.mean(row) <= 150 and np.mean(new_img.transpose(1, 0, 2)[j-1]) > 150:
                start = j
            elif np.mean(row) > 150 and np.mean(new_img.transpose(1, 0, 2)[j-1]) <= 150 and j != 0:
                if j - start > 100:
                    end = j
                break
        new_img = new_img.transpose(1, 0, 2)[start:end].transpose(1, 0, 2)
        imgs_rotated.append(new_img)

    return imgs_rotated

def smooth(imgs: list[np.ndarray]) -> list[np.ndarray]:
    """
    Smooth image and remove artefacts from rotation

    Args:
        imgs: list of rotated images

    Returns:
        list of smoothed images
    """
    imgs_smoothed = []
    for img in imgs:
        new_img = img
        for k, row in enumerate(img):
            for j, pixel in enumerate(row):
                if np.mean(pixel) == 255:
                    try:
                        neighbours = np.array([row[j-1], row[j+1], img[k-1][j], img[k+1][j]])
                    except IndexError:
                        continue
                    new_img[k][j] = np.mean(neighbours, 0)
        imgs_smoothed.append(new_img)
    return imgs_smoothed

def crop_rot(path: str) -> list[np.ndarray]:
    """
    Crop and rotate image properly
    
    Args:
        path: fullpath to image

    Returns:
        list of cropped, rotated and smoothed images as numpy arrays
    """
    all_image = import_tif(path)
    all_images_sliced = slice_image(all_image)
    all_images_rotated = rotate(all_images_sliced)
    all_images_smoothed = smooth(all_images_rotated)
    return all_images_smoothed

if __name__ == "__main__":
    print("Importing image...")
    image = import_tif()
    print("Slicing image...")
    images_sliced = slice_image(image)
    print("Rotating images...")
    images_rotated = rotate(images_sliced)
    print("Smoothing images...")
    images_smoothed = smooth(images_rotated)
    print("Showing images...")
    for i, img_smoothed in enumerate(images_smoothed):
        im = Image.fromarray(img_smoothed.astype(np.uint8))
        im.show()
        im.save(f'Cropped_Rotated_Images/stack_image_{i+1}.tif')

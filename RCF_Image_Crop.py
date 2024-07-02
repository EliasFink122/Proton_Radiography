# -*- coding: utf-8 -*-
"""
Created on Mon Jul 01 2024

@author: Elias Fink (elias.fink22@imperial.ac.uk)

Cropping and rotating RCF stack images from initial scan
"""
import numpy as np
from PIL import Image

def import_tif() -> np.ndarray:
    """
    Import tif image

    Returns:
        numpy array of pixel rgb values with higher contrast
    """
    im = Image.open('sh1_st1.tif')
    imarray = np.array(im)

    for i, row in enumerate(imarray):
        for j, pixel in enumerate(row):
            if np.mean(pixel) > 200:
                imarray[i][j] = [255, 255, 255]
    return imarray

def slice_image(img: np.ndarray) -> list[np.ndarray]:
    """
    Slices full image roughly into sections of stack images

    Args:
        img: pixels of full input image

    Returns:
        array of sections in order of stack
    """
    slices_x = []
    starts = []
    ends = []
    for i, row in enumerate(img):
        if np.mean(row) <= 240 and np.mean(img[i-1]) > 240:
            starts.append(i)
        elif np.mean(row) > 240 and np.mean(img[i-1]) <= 240:
            ends.append(i)
    for i, start in enumerate(starts):
        if ends[i] - start > 100:
            slices_x.append(img[start:ends[i]])

    imgs_sliced = []
    for slice_x in slices_x:
        slice_x = np.transpose(slice_x, (1, 0, 2))
        starts = []
        ends = []
        for i, row in enumerate(slice_x):
            if np.mean(row) <= 254 and np.mean(slice_x[i-1]) > 254:
                starts.append(i)
            elif np.mean(row) > 254 and np.mean(slice_x[i-1]) <= 254:
                ends.append(i)
        for i, start in enumerate(starts):
            if ends[i] - start > 100:
                imgs_sliced.append(slice_x[start:ends[i]].transpose(1, 0, 2))

    for i, img_sliced in enumerate(imgs_sliced):
        start = 0
        end = len(img_sliced) - 1
        for j, row in enumerate(img_sliced):
            if np.mean(row) <= 254 and np.mean(img_sliced[j-1]) > 254:
                start = j
            elif np.mean(row) > 254 and np.mean(img_sliced[j-1]) <= 254 and j != 0:
                end = j
                break
        imgs_sliced[i] = img_sliced[start:end]
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
        edge_x = np.array([])
        edge_y = np.array([])
        for i, row in enumerate(img):
            for j, pixel in enumerate(row):
                if np.mean(pixel) <= 200:
                    edge_x = np.append(edge_x, j)
                    edge_y = np.append(edge_y, i)
                    break

        for i, p_x in enumerate(edge_x):
            if np.abs(p_x - edge_x[i]) >= 5:
                edge_x = edge_x[i:]
                edge_y = edge_y[i:]

        p1 = np.array([edge_x[int(0.2 * len(edge_x))],
                       edge_y[int(0.2 * len(edge_y))]])
        p2 = np.array([edge_x[int(0.8 * len(edge_x))],
                       edge_y[int(0.8 * len(edge_y))]])

        theta = np.arctan((p2[0] - p1[0])/(p2[1] - p1[1]))

        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta), np.cos(theta)]])
        new_img = np.full(np.shape(img), 255)
        for i, row in enumerate(img):
            for j, pixel in enumerate(row):
                if np.mean(pixel) != 255:
                    old_xy = np.array([j - len(img[0])/2, i - len(img)/2])
                    new_xy = np.dot(rotation_matrix, old_xy)
                    try:
                        new_j = int(new_xy[0] + len(row)/2)
                        new_i = int(new_xy[1] + len(img)/2)
                        if new_j < 0 or new_i < 0:
                            continue
                        new_img[new_i, new_j] = pixel
                    except IndexError:
                        continue
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
        for i, row in enumerate(img):
            for j, pixel in enumerate(row):
                if np.mean(pixel) == 255:
                    try:
                        neighbours = np.array([row[j-1], row[j+1], img[i-1][j], img[i+1][j]])
                    except IndexError:
                        continue
                    new_img[i][j] = np.mean(neighbours, 0)
        imgs_smoothed.append(new_img)
    return imgs_smoothed

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
        im.save(f'Output/stack_image_{i+1}.tif')

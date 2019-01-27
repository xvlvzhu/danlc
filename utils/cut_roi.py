import cv2
import numpy as np
import tifffile as tiff
import os


def scale_percentile(matrix):
    w, h, d = matrix.shape
    matrix = np.reshape(matrix, [w * h, d]).astype(np.float64)
    # Get 2nd and 98th percentile
    mins = np.percentile(matrix, 1, axis=0)
    maxs = np.percentile(matrix, 99, axis=0) - mins
    matrix = (matrix - mins[None, :]) / maxs[None, :]
    matrix = np.reshape(matrix, [w, h, d])
    matrix = matrix.clip(0, 1)
    return matrix


def tiff2ImgSclice(file_path, save_path, img_size=256, prefix=''):
    im_1 = tiff.imread(file_path).transpose([0, 1, 2])
    print(im_1.shape)
    # img_size = 512 # 15106/ 256 =59...2  5106/256=19..284
    file_dir = os.path.split(file_path)[0]
    file_name = os.path.split(file_path)[1].split(".")[0]
    for i in range(int(len(im_1) / img_size)):  # last 284
        for j in range(int(len(im_1[0]) / img_size)):  # last 2 too small, drop one
            im_name = prefix + file_name + '_' + str(i) + '_' + str(j) + '_' + str(img_size) + '.tif'
            print(im_name)
            print(os.path.join(file_dir, im_name))
            tiff.imsave(os.path.join(save_path, im_name),
                        im_1[i * img_size:i * img_size + img_size, j * img_size:j * img_size + img_size, :])


if __name__ == '__main__':
    tiff2ImgSclice('./data')

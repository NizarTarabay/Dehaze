'''
This code is the implementation of the "Guided Image FilteringKaiming He1, Jian Sun2, and Xiaoou Tang" paper
and "Single Image Haze RemovalUsing Dark Channel Prior Kaiming He, Jian Sun, and Xiaoou Tang,Fellow,IEEE"
with a modification in the atmospheric light computation
(the dark channel prior is not a good prior for the sky regions to compute the atmospheric light)
code reference:
dehaze code: modified from: https://github.com/anhenghuang/dehaze
guided filter: modified from http://kaiminghe.com/eccv10/
modification: python implementation
'''

import cv2
import numpy as np
import argparse

# ----------------------------- #
# --- Find the dark channel --- #
# ----------------------------- #
def darkChannel(image, size = 15):
    '''
    Extract the dark channel from the image
    image: ndarray
        The imput image
    size: int
        The patch size
    :return: ndarray
        the dark channel
    '''
    # split the image into the blue, green, and red channel
    blue, green, red = cv2.split(image)

    # get the min B G R value for each pixel
    minImg = cv2.min(red, cv2.min(green, blue))

    # get a size x size rectangular kernel i.e., the local patch
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))

    # Looking for the minimum inside the local patch:
    # apply the erode morphological function on the minimum channel image
    darkChanelImage = cv2.erode(minImg, kernel)

    return darkChanelImage

# ------------------------- #
# --- Atmospheric light --- #
# ------------------------- #
# Assumption: the atmospheric light is in the top 1% of the image
def atmosphericLight(img, percent = 0.01):
    '''

    :param img: ndarray
        input image
    :param percent: float
        The top section of the image
    :return: float
        The average of the average intensity og the top 1% of the image
    '''
    # Get the average intensity of a pixel
    mean_perpix = np.mean(img, axis = 2).reshape(-1)
    # Get the the first 1 percent of the top pixels and compute their mean value
    mean_topper = mean_perpix[:int(img.shape[0] * img.shape[1] * percent)]
    return np.mean(mean_topper)

# ----------------------------------- #
# --- Estimating the Transmission --- #
# ----------------------------------- #
# equation 11 in :
# "Single Image Haze Removal using Dark Channel Prior Kaiming He, Jian Sun, and Xiaoou Tang,Fellow,IEEE"
def estimateTransmission(img, atmo, w = 0.95):

    x = img / atmo
    t = 1 - w * darkChannel(x, 15)
    return t

# --------------------- #
# --- Guided filter --- #
# --------------------- #
# Transmission Map Enhancement: implementation of the guided filter
# for smoothening and detail enhancement of the image
def guidedFilter(p, i, r, e):
    """
    :param p: input image
    :param i: guidance image
    :param r: radius
    :param e: regularization
    :return: filtering output q
    """
    # Image smoothing
    meanGuidanceImage = cv2.boxFilter(i, cv2.CV_64F, (r, r))
    meanInputImage = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    correctionGuidanceImage = cv2.boxFilter(i * i, cv2.CV_64F, (r, r))
    correctionGuidanceInputImage = cv2.boxFilter(i * p, cv2.CV_64F, (r, r))

    varI = correctionGuidanceImage - meanGuidanceImage ** 2
    covIp = correctionGuidanceInputImage - meanGuidanceImage * meanInputImage  # this is the covariance of (i, p) in each local patch

    # "the relationship among I, p, and q given by (5), (6), and (8) are indeed in the form of image filtering"
    a = covIp / (varI + e)  # Eqn. (5) in "Guided Image Filtering Kaiming He1, Jian Sun2, and Xiaoou Tang" paper;
    b = meanInputImage - a * meanGuidanceImage  # Eqn. (6) in "Guided Image Filtering Kaiming He1, Jian Sun2, and Xiaoou Tang" paper

    meanA = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    meanB = cv2.boxFilter(b, cv2.CV_64F, (r, r))

    q = meanA * i + meanB  # Eqn. (8) in "Guided Image Filtering Kaiming He1, Jian Sun2, and Xiaoou Tang" paper
    return q

# ----------------------------- #
# --- Result Reconstruction --- #
# ----------------------------- #
def dehaze(path, output = None):
    im = cv2.imread(path)
    # im = cv2.resize(im, (int(480), int(270)))
    img = im.astype('float64') / 255
    img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype('float64') / 255

    atmo= atmosphericLight(img)
    # atom = Nmaxelements(darkChannel(img, size=1), percent=0.1)
    trans = estimateTransmission(img, atmo)
    transGuided = guidedFilter(trans, img_gray, 20, 0.0001)
    transGuided = cv2.max(transGuided, 0.25)

    # Result reconstruction
    result = np.empty_like(img)
    for i in range(3):
        result[:, :, i] = (img[:, :, i] - atmo[i]) / transGuided + atmo[i]

    cv2.imshow("Source", img)
    cv2.imshow("Result", result)
    cv2.imshow("Trans", trans)
    cv2.imshow("Dark Channel", darkChannel(img, size=3))
    if output is not None:
        cv2.imwrite(output, result * 255)
    cv2.waitKey()



parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input')
parser.add_argument('-o', '--output')
args = parser.parse_args()


if __name__ == '__main__':
    if args.input is None:
        dehaze('image/t012_frame1.png')
    else:
        dehaze(args.input, args.output)
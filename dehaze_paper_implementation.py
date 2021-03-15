'''
This code is the implementation of the "Guided Image FilteringKaiming He1, Jian Sun2, and Xiaoou Tang" paper
and "Single Image Haze RemovalUsing Dark Channel Prior Kaiming He, Jian Sun, and Xiaoou Tang,Fellow,IEEE"
code reference:
-Modified from: https://github.com/anhenghuang/dehaze
Modification: (1) The atmospheric light computation is modified to the one implemented in
"Single Image Haze RemovalUsing Dark Channel Prior Kaiming He, Jian Sun, and Xiaoou Tang,Fellow,IEEE"
(2) The channel order is modified from RGB to GBR
(3) Reference to equations in the two paper above
guided filter: modified from http://kaiminghe.com/eccv10/
modification: python implementation
'''

import cv2
import numpy as np
import argparse
import math

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

# ------------------------------------ #
# --- Atmospheric light estimation --- #
# ------------------------------------ #
# Paper implementation
def atmLight(im,dark):
    [h,w] = im.shape[:2]
    imsz = h*w
    numpx = int(max(math.floor(imsz/1000), 1))
    darkvec = dark.reshape(imsz)
    imvec = im.reshape(imsz,3)

    indices = darkvec.argsort()
    indices = indices[imsz-numpx::]

    atmsum = np.zeros([1,3])
    for ind in range(1,numpx):
       atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx
    return A

# ----------------------------------- #
# --- Estimating the Transmission --- #
# ----------------------------------- #
# Equation 11 in Single Image Haze Removal in "Dark Channel Prior Kaiming He, Jian Sun, and Xiaoou Tang,Fellow,IEEE"
def estimateTransmission(img, atmo, w = 0.95):

    x = img / atmo
    t = 1 - w * darkChannel(x, 15)
    return t

# --------------------- #
# --- Guided filter --- #
# --------------------- #
def guidedFilter(p, i, r, e):
    '''
    Transmission Map Enhancement: implementation of the guided filter
    for smoothening and detail enhancement of the image
    :param p: input image
    :param i: guidance image
    :param r: radius
    :param e: regularization
    :return: filtering output q
    '''
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
    img = im.astype("float64") / 255
    img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype('float64') / 255

    atmo = atmLight(img, darkChannel(img, size=15))
    trans = estimateTransmission(img, atmo)
    trans_guided = guidedFilter(trans, img_gray, 20, 0.0001)
    trans_guided = cv2.max(trans_guided, 0.25)

    # Result reconstruction
    result = np.empty_like(img)
    for i in range(3):
        result[:, :, i] = (img[:, :, i] - atmo[0, i]) / trans_guided + atmo[0, i]

    cv2.imshow("Source", img)
    cv2.imshow("Result", result)
    cv2.imshow("Trans", trans)
    cv2.imshow("Dark Channel", darkChannel(img, size=3))
    cv2.waitKey()
    if output is not None:
        cv2.imwrite(output, result * 255)


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input')
parser.add_argument('-o', '--output')
args = parser.parse_args()


if __name__ == '__main__':
    if args.input is None:
        dehaze('image/frameOUT002_new.jpg')
    else:
        dehaze(args.input, args.output)
# Author: Goh Zhao Yang

# Standard scientific Python imports
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics

# Param for de-skewing
affine_flags = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR
SZ=20
# Param for HOG extraction
bin_n = 16

# The digits dataset (1. training 2. show ori img 3. test individual img 4. test bulk img)
digits = datasets.load_digits()
real_digit = datasets.load_digits()
test_digit = datasets.load_digits()
test_digit_2 = datasets.load_digits()

#
# De-skew operation
def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02'])< 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew],[0,1,0]])
    img = cv2.warpAffine(img,M,(SZ,SZ), flags=affine_flags)
    return img

# Calculate HOG feature vector
def hog(img):
    gx = cv2.Sobel(img, 6, 1, 0)
    gy = cv2.Sobel(img, 6, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n * ang / (2 * np.pi))  # quantizing binvalues in (0...16)
    bin_cells = bins[:10, :10], bins[10:, :10], bins[:10, 10:], bins[10:, 10:]
    mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)  # hist is a 64 bit vector
    return hist

# Count number of images in dataset
n_samples = len(digits.images)
data = []

# Looping to append HOG into data tuple
for i in range(1,(n_samples/2)):
    deskewed = deskew(digits.images[i])
    hogdata=hog(deskewed)
    data.append(hogdata)

# Initialize predicted as empty tuple
predicted = []

######################### TESTING BULK #######################################
for a in range(n_samples):
    max = 0
    max_i = 1
    for b in range(len(data)):
        temp = cv2.compareHist(np.float32(np.array(hog(deskew(test_digit_2.images[a])))),np.float32(np.array(data[b])),cv2.cv.CV_COMP_CORREL )
        if abs(temp)>max:
            max = abs(temp)
            max_i = digits.target[b+1]
    predicted.append(max_i)
expected = test_digit_2.target[:(n_samples)]

print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
######################### END TESTING 1 ###############################################

######################### START TESTING 2 #############################################
# USER INPUT
indexFile = int(raw_input('Test image number?'))

# DECLARE INDEX WITH HIGHEST CORRELATION
max_index = 1
# TEST IS IMAGES
test = test_digit.images
# PERFORM DE-SKEWING
test_deskewed = deskew(test[indexFile])
# GET HOG FEATURE VECTOR
test_hogdata = hog(test_deskewed)
# FOR LOOP TO COMPARE TESTING IMAGE HOG WITH THE STORED HOG
for i in range(len(data)):
    temp = cv2.compareHist(np.float32(np.array(test_hogdata)),np.float32(np.array(data[i])),cv2.cv.CV_COMP_CORREL)
    # SWAP TO GET MAXIMUM VALUE
    if abs(temp)>max:
        max = abs(temp)
        max_index = i+1

############################ TESTING 2 END #######################################################

# SHOW ORIGINAL IMAGE
plt.imshow(real_digit.images[indexFile], cmap=plt.cm.gray_r, interpolation='nearest')
# SHOW RESULT
plt.title("Predicted : %i" %real_digit.target[max_index] )
plt.axis('off')
plt.show()






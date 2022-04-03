from PIL import Image, ImageOps
import os
import cv2 as cv
from tqdm import tqdm
import numpy as np

imgnames = os.listdir(".")
i = 0

import random
listdata = [i for i in range(2056)]
train_filenames = random.sample(listdata,int(len(listdata)*0.75))
test_filesnames = [k for k in listdata if k not in train_filenames]
random.shuffle(test_filesnames)

for filename in train_filenames:
    #print("./Dataset4K/Grayscale/"+str(filename)+".png")
    train_imgG = np.array(cv.imread("./Grayscale/"+str(filename)+".png"))
    train_imgC = np.array(cv.imread("./Colour/"+str(filename)+".png"))
    train_imgC = np.pad(train_imgC,[(56,56),(0,0),(0,0)],mode="mean")
    train_imgG = np.pad(train_imgG,[(56,56),(0,0),(0,0)],mode="mean")

    #print(train_imgC.shape)
    #train_imgG = np.append(train_imgG,train_imgG,axis = 2)
    #train_imgG = np.append(train_imgG,train_imgG,axis = 2)
    train_imgC = np.append(train_imgG,train_imgC,axis = 1)
    cv.imwrite("./Train/"+str(filename)+".png",train_imgC)
print("TEST")
for filename in test_filesnames:
    test_imgG = np.asarray(cv.imread("./Grayscale/"+str(filename)+".png"))
    test_imgC = np.asarray(cv.imread("./Colour/"+str(filename)+".png"))
    test_imgC = np.pad(test_imgC,[(56,56),(0,0),(0,0)],mode="mean")
    test_imgG = np.pad(test_imgG,[(56,56),(0,0),(0,0)],mode="mean")
    #test_imgG = np.append(test_imgG,test_imgG,axis = 2)
    #test_imgG = np.append(test_imgG,test_imgG,axis = 2)
    test_imgC = np.append(test_imgG,test_imgC,axis = 1)

    cv.imwrite("./Test/"+str(filename)+".png",test_imgC)


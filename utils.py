import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.sparse import random
from sklearn.utils import shuffle
import numpy as np
import matplotlib.image as mpimg
from imgaug import augmenters as iaa
import cv2 
import random

def getName(filePath):
    return filePath.split("\\")[-1]


def importDataInfo(path):
    columns = ["Center", "Left", "Right", "Steering", "Throttle", "Brake", "Speed"]
    data = pd.read_csv(os.path.join(path, "driving_log.csv"), names=columns)
    data["Center"] = data["Center"].apply(getName)
    print(data.head())
    print("Total images Imported: " , data.shape[0])
    return data


# Helper funtions
def balanceData(data, display=True):
    nBins = 31
    samplesPerBin = 1000
    hist, bins = np.histogram(data["Steering"], nBins)
    if display:
        center = (bins[:-1] + bins[1:]) * 0.5
        plt.bar(center, hist, width=0.06)
        plt.plot((-1,1), (samplesPerBin, samplesPerBin))
        plt.show()

    removeIndexList = []
    for j in range(nBins):
        binDataList = []
        for i in range(len(data["Steering"])):
            if data["Steering"][i] >= bins[j] and data["Steering"][i] <= bins[j+1]:
                binDataList.append(i)
        binDataList = shuffle(binDataList)
        binDataList = binDataList[samplesPerBin:]
        removeIndexList.extend(binDataList)
    print("Removed Images: " , len(removeIndexList))
    balanced_data = data.copy()
    balanced_data.drop(data.index[removeIndexList], inplace=True)
    print("Remaining Images: " , len(data))
    if display:
        hist, bins = np.histogram(balanced_data["Steering"], nBins)
        plt.bar(center, hist, width=0.06)
        plt.plot((-1,1), (samplesPerBin, samplesPerBin))
        plt.show()

    return balanced_data


def loadData(path, data):
    imagesPath = []
    steering = []

    for i in range(len(data)):
        indexedData = data.iloc[i]
        imagesPath.append(os.path.join(path, "IMG", indexedData[0]))
        steering.append(float(indexedData[3]))
    
    imagesPath = np.asarray(imagesPath)
    steering = np.asarray(steering)
    return imagesPath, steering


def augmentImage(imagePath, steering):
    img = mpimg.imread(imagePath)
    if (np.random.rand() < 0.5):
        ## Pan
        pan = iaa.Affine(translate_percent={"x":(-.1, 0.1), "y":(-0.1, 0.1)})
        img = pan.augment_image(img)
    if (np.random.rand() < 0.5):
        ## Zoom
        zoom =iaa.Affine(scale=(1,1.2))
        img = zoom.augment_image(img)
    if (np.random.rand() < 0.5):
        ## Brigtness
        brightness = iaa.Multiply((0.3, 1.2))
        img = brightness.augment_image(img)
    if (np.random.rand() < 0.5):
        ## Flip
        img = cv2.flip(img, 1)
        steering = -steering
        
    return img, steering

def preProcessing(img):
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3,3), 0)
    img = cv2.resize(img, (200,66))
    img = img/255

    return img

def batchGenerator(imagesPath, steeringList, batchSize, trainFlag):
    while True:
        imageBatch = []
        steeringBatch = []

        for i in range(batchSize):
            index = random.randint(0, len(imagesPath)-1)
            if trainFlag:
                img, steering = augmentImage(imagesPath[index], steeringList[index])
            else:
                img = mpimg.imread(imagesPath[index])
                steering = steeringList[index]
            img = preProcessing(img)
            imageBatch.append(img)
            steeringBatch.append(steering)

        yield (np.asarray(imageBatch), np.asarray(steeringBatch))

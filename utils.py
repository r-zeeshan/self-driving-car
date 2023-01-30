import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def getName(filePath):
    return filePath.split("\\")[-1]


def importDataInfo(path):
    columns = ["Center", "Left", "Right", "Steering", "Throttle", "Brake", "Speed"]
    data = pd.read_csv(os.path.join(path, "driving_log.csv"), names=columns)
    data["Center"] = data["Center"].apply(getName)
    print(data.head())
    print("Total images Imported: " , data.shape[0])
    return data


def balanceData(data, display=True):
    nBins = 31
    samplesPerBin = 500
    hist, bins = np.histogram(data["Steering"], nBins)
    center = (bins[:-1] + bins[1:]) * 0.5
    plt.bar(center, hist, width=0.06)
    plt.show()


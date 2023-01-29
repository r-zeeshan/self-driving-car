import pandas as pd
import numpy as np
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



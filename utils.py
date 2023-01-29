import pandas as pd
import numpy as np
import os

def importDataInfo(path):
    columns = ["Center", "Left", "Right", "Steering", "Throttle", "Brake", "Speed"]
    # data = pd.read_csv(os.path.join(path, "driving_log.csv"), names=columns)
    data = pd.read_csv("/content/self-driving-car/SimulationData/driving_log.csv", names=columns)
    print(data.head())
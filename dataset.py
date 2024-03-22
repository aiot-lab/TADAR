import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm
import os
import pickle


class Dataset():
    def __init__(self, datapaths) -> None:
        self.ira_matrix = []
        self.ambient_temperature = []
        self.timestamps = []
        self.GT_bbox = []
        self.GT_depth = []
        self.GT_range = []
        for datapath in datapaths:
            file = open(datapath, 'rb')
            data = pickle.load(file)
            matrix = data['ira_temperature_matrix']
            at = data['ira_ambient_temperature']
            ts = data['timestamps']
            bbox = data['GT_bbox']
            depth = data['depth']
            range_ = data['range']
            file.close()
            
            self.ira_matrix += matrix
            self.ambient_temperature += at
            self.timestamps += ts
            self.GT_bbox += bbox
            self.GT_depth += depth
            self.GT_range += range_
    
    def GetSample(self, index):
        return self.ira_matrix[index], self.ambient_temperature[index], self.timestamps[index], self.GT_bbox[index], self.GT_depth[index], self.GT_range[index]
    
    def GetAllSamples(self):
        return self.ira_matrix, self.ambient_temperature, self.timestamps, self.GT_bbox, self.GT_depth, self.GT_range
    
    def len(self):
        return min(len(self.ira_matrix), len(self.ambient_temperature), len(self.timestamps), len(self.GT_bbox), len(self.GT_depth), len(self.GT_range))

if __name__ == "__main__":
    datapaths = [
        'Dataset/Hall_0_sensor_4.pickle',
    ]

    dataset = Dataset(datapaths)
    print(dataset.len())

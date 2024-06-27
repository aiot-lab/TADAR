import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# from skimage.segmentation import clear_border
from skimage import measure
from skimage.measure import label,regionprops
from scipy import ndimage as ndi
from tqdm import tqdm
import ast
import numba as nb
from skimage.filters import threshold_multiotsu
from scipy import signal
import torch.nn as nn
import torch
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn import neighbors
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor



# zone map only support the MLX90640-110 sensor
def GetZoneMap():
    zones_map = np.zeros((24, 32))
    for row in range(24):
        for col in range(32):
            if (row > 3 and row < 20) and (col >7 and col < 24):
                zones_map[row, col] = 1
            elif col < 6 and row < 6-col:
                zones_map[row, col] = 3
            elif col < 6 and row >17 and (row-17) > col:
                zones_map[row, col] = 3
            elif col > 25 and row < (col-25):
                zones_map[row, col] = 3
            elif col>25 and row> 17 and ((col-25) > 6 -(row-17)):
                zones_map[row, col] = 3
            else:
                zones_map[row, col] = 2
    return zones_map

# return chessboard reading pattern as a 2D numpy arrray
def GetChessboard(shape):
    chessboard = np.indices(shape).sum(axis=0) % 2
    chessboard_inverse = np.where((chessboard==0)|(chessboard==1), chessboard^1, chessboard)
    return chessboard, chessboard_inverse

# Checking the type of the subpage.
def SubpageType(mat, chessboard):
        subpage0 = mat * chessboard
        if np.sum(subpage0) < 1:
            return 1
        return 0

# Interpolating the subpage into a complete frame by using the bilinear interpolating method with window size at 3x3.
def SubpageInterpolating(subpage):
    shape = subpage.shape
    mat = subpage.copy()
    for i in range(shape[0]):
        for j in range(shape[1]):
            if mat[i,j] > 0.0:
                continue
            num = 0
            try:
                top = mat[i-1,j]
                num = num+1
            except:
                top = 0.0
            
            try:
                down = mat[i+1,j]
                num = num+1
            except:
                down = 0.0
            
            try:
                left = mat[i,j-1]
                num = num+1
            except:
                left = 0.0
            
            try:
                right = mat[i,j+1]
                num = num+1
            except:
                right = 0.0
            mat[i,j] = (top + down + left + right)/num
    return mat


############ This part is the preprocessing pipline:
# first component: Deal with the outliers in the IRA data
class Preprocess():
    def __init__(self,chessboard):
        self.chessboard = chessboard
        self.chessboard_inverse = np.where((chessboard==0)|(chessboard==1), chessboard^1, chessboard)
         
    def Outlier1TypeDelete(self, mat):
        subpage0 = mat * self.chessboard
        subpage1 = mat* self.chessboard_inverse
        num_pixels_subpage = int(np.sum(self.chessboard))
        if np.sum(subpage0) > 300*num_pixels_subpage or np.sum(subpage1) > 300*num_pixels_subpage:
            return 0     # we need to discard this sample
        return 1         # we can keep this sample
    
    def Outlier2TypeElimilate(self, mat):
        mat_copy = mat.copy()
        outlier_position = np.where(mat>300)
        rows = outlier_position[0]
        cols = outlier_position[1]
        stats = 1      # there is no outliers in this sample
        for index,row in enumerate(rows):
            i = row
            j = cols[index]
            num = 0
            stats = 2    # there exists outliers in this sample
            try:
                topleft = mat_copy[i-1,j-1]
                num = num+1
            except:
                topleft = 0.0
            
            try:
                bottomleft = mat_copy[i+1,j-1]
                num = num+1
            except:
                bottomleft = 0.0
            
            try:
                topright = mat_copy[i-1,j+1]
                num = num+1
            except:
                topright = 0.0
            
            try:
                bottomright = mat_copy[i+1,j+1]
                num = num+1
            except:
                bottomright = 0.0
            mat_copy[i,j] = (topleft + bottomleft + topright + bottomright)/num
        return mat_copy, stats
    
    def Forward(self, mat):
        """
        stats:
            0: discard this sample.
            1: all pixels are perfect.
            2: there are some outliers in this sample but fixed by interpolating.
        """
        mat = mat.copy()
        stats = self.Outlier1TypeDelete(mat)
        if stats:
            mat, stats = self.Outlier2TypeElimilate(mat)
        return mat, stats

# second component: Change the IRA data to a Image-like version
class BaseProcess():
    def __init__(self, expansion_coefficient = 20, temperature_upper_bound = 34) -> None:
        """Initailization

        Args:
            expansion_coefficient (int, optional): the shape expansion ratio of the received temperature matrix. Defaults to 20.
            temperature_upper_bound (int, optional): the highest temperature that is considered. Defaults to 33.
        """
        self.expansion_coefficient = expansion_coefficient
        self.temperature_upper_bound = temperature_upper_bound
    
    def BandPass(self, sensor_mat, sensor_at):
        matrix1 = np.where(sensor_mat < self.temperature_upper_bound, sensor_mat, sensor_at)
        return matrix1
    
    def Normalize(self,sensor_mat):
        min_v = np.min(sensor_mat)
        matrix2 = (sensor_mat - min_v) / (self.temperature_upper_bound-min_v)
        return matrix2
    
    def ChangeScale(self,sensor_mat):
        matrix3 = np.array(sensor_mat * 255, dtype= np.uint8)
        return matrix3
    
    def Interpolate(self, sensor_mat):
        original_shape = sensor_mat.shape
        matrix4 = cv2.resize(sensor_mat, 
                             (original_shape[1]* self.expansion_coefficient,original_shape[0]* self.expansion_coefficient), 
                             interpolation=cv2.INTER_LINEAR
                             )
        return matrix4, original_shape
    
    def Forward(self, sensor_mat, sensor_at):
        """processing

        Args:
            sensor_mat (numpy.array): the temperature matrix from IRA sensor
            sensor_at (float): the detected ambient temperature from IRA sensor

        Returns:
            (numpy.array, tuple): the processed temperature matrix, the original shape of the temperature matrix.
        """
        matrix1 = self.BandPass(sensor_mat, sensor_at)
        matrix2 = self.Normalize(matrix1)
        matrix3 = self.ChangeScale(matrix2)
        matrix4, original_shape = self.Interpolate(matrix3)
        return matrix4, original_shape

# The pipeline of the preprocessing 
class PrePipeline():
    def __init__(self,expansion_coefficient = 10, temperature_upper_bound = 34, buffer_size = 10, data_shape = (24,32)) -> None:
        self.expansion_coefficient = expansion_coefficient
        self.temperature_upper_bound = temperature_upper_bound
        self.buffer = []
        self.buffer_size = buffer_size
        self.chessboard, _ = GetChessboard(data_shape)
        self.preprocessor = Preprocess(self.chessboard)
        self.basic_process = BaseProcess(expansion_coefficient, temperature_upper_bound)
    
    def PreProcessing(self, sensor_mat):
        # subpage, stats = self.preprocessor.Forward(np.flip(sensor_mat,0))
        subpage, stats = self.preprocessor.Forward(sensor_mat)
        if stats == 0:
            print("This subpage is invalid!")
            return 0, subpage
        if stats == 2:
            print("This subpage contains outliers that have been replaced by using average values of their nearby elements.")
        frame = SubpageInterpolating(subpage)     # interpolate the subpage into complete frame
        return frame, subpage
    
    def Forward(self, sensor_mat, sensor_at):
        frame, subpage = self.PreProcessing(sensor_mat)
        if not isinstance(frame, (np.ndarray)):
            return 0, 0, 0
        subpage_type = SubpageType(subpage, self.chessboard)
        ira_img, _ = self.basic_process.Forward(frame, sensor_at)
        
        original_shape = frame.shape
        bandpass_frame = self.basic_process.BandPass(frame, sensor_at)
        ira_mat = cv2.resize(bandpass_frame, 
                            (original_shape[1]* self.expansion_coefficient,original_shape[0]* self.expansion_coefficient), 
                            interpolation=cv2.INTER_LINEAR
                            )
        
        self.buffer.append((ira_img, subpage_type))
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
            
        return ira_img, subpage_type, ira_mat
############# End of the preprocessing


############# The segmentation part
# first component: considering the spatial information
class DetectingProcess():
    def __init__(self, expansion_coefficient = 20, valid_region_area_limit = 5) -> None:
        """Initailization

        Args:
            expansion_coefficient (int, optional): the shape expansion ratio of the received temperature matrix. Defaults to 20.
            valid_region_area_limit (int, optional): the area of the smallest region that is considered. Defaults to 20.
        """
        self.expansion_coefficient = expansion_coefficient
        self.valid_region_area_limit = valid_region_area_limit

    def AdaptiveBinary(self,sensor_mat,maximum, box_size):
        """
        https://docs.opencv.org/4.7.0/d7/d1b/group__imgproc__misc.html#ga72b913f352e4a1b1b397736707afcde3
        """
        mask = cv2.adaptiveThreshold(sensor_mat, maximum,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, box_size, 0)
        return mask
    
    def BorderRemove(self, mask):
        # removing regions connected to up, left, and right borders in the mask.
        mask_labeled_temp = label(mask)
        mask2 = np.zeros_like(mask)
        rps_temp = regionprops(mask_labeled_temp)
        for i in range(len(rps_temp)):
            corordinates = rps_temp[i].coords.T
            if mask.shape[0]-1 in corordinates[0]: # regions connected to the bottom border.
                mask2[tuple(corordinates)] = 255
            elif (0 in corordinates[1]) or 0 in corordinates[0]  or (mask.shape[1]-1 in corordinates[1]):  # regions connected to left,right,up borders.
                pass
            else:                                              # regions inside the image
                mask2[tuple(corordinates)] = 255
        return mask2
    
    def BlobRemove(self, mask, pixel_limit):
        # discarding the regions with areas less than pixels limit. 
        mask2_labeled = label(mask)
        rps = regionprops(mask2_labeled)
        areas = [r.area for r in rps]
        mask3 = np.zeros_like(mask)
        for i in range(len(areas)):
            if areas[i] > pixel_limit:
                mask3[tuple(rps[i].coords.T)] = 255
        return mask3
    
    def RegionDivid(self, mask):
        masks = []
        copy_mask = mask.copy()
        label_mask = label(copy_mask)
        rps = regionprops(label_mask)
        for i in range(len(rps)):
            corordinates = rps[i].coords.T
            temp_mask = np.zeros_like(mask)
            temp_mask[tuple(corordinates)] = 1
            masks.append(temp_mask)
        return masks
    
    def TopKRegion(self, mask, topk):
        # keep top k regions
        mask_labeled = label(mask)
        rps = regionprops(mask_labeled)
        areas = [r.area for r in rps]
        idx = np.argsort(areas)[::-1]
        if topk < len(idx):
            selected_index = idx[:topk]
        else:
            selected_index = idx
        topk_mask = np.zeros_like(mask)
        for index in selected_index:
            topk_mask[tuple(rps[index].coords.T)] = 255
        return topk_mask
    
    def RegionColored(self,mask):
        # Given different region with different values
        mask_labeled = label(mask)
        rps = regionprops(mask_labeled)
        num_regions = len(rps)
        if num_regions == 0:
            return mask
        values = np.linspace(70,190,num_regions)
        re_mask = np.zeros_like(mask)
        for i in range(len(rps)):
            re_mask[tuple(rps[i].coords.T)] = values[i]
        return re_mask
    
    def TemperatureDistributionCalc(self, frame, mask):
        temp_frame = frame.copy()
        temp_mask = np.where(mask>0.2, 1, 0)
        mask_frame = temp_frame*temp_mask
        distri_x = (np.sum(mask_frame, axis = 0))
        
        bins_x = np.linspace(0.0,float(len(distri_x)), len(distri_x))
        return distri_x, bins_x, mask_frame
    
    def CandidateCount_x(self, hist, num_user_bound):
        nonzero_index = np.nonzero(hist)
        first_nonzero_index = nonzero_index[0][0]
        last_nonzero_index = nonzero_index[0][-1]
        hist_no_zero = hist[first_nonzero_index:last_nonzero_index+1]
        hist_no_zero = hist_no_zero/ np.max(hist_no_zero)
        height = np.mean(hist_no_zero)
        fs = len(hist_no_zero)
        fc = num_user_bound  
        try:
            w = fc / (fs / 2) 
            b, a = signal.butter(5, w, 'low')
            filtered = signal.filtfilt(b, a, hist_no_zero)
        except:
            try:
                fc = num_user_bound / 2
                w = fc / (fs / 2) 
                b, a = signal.butter(5, w, 'low')
                filtered = signal.filtfilt(b, a, hist_no_zero)
            except:
                fc = num_user_bound / 4
                w = fc / (fs / 2) 
                b, a = signal.butter(5, w, 'low')
                filtered = signal.filtfilt(b, a, hist_no_zero)
        peaks, _ = signal.find_peaks(filtered, height, prominence=0.1)
        peaks_in_hist = peaks + first_nonzero_index
        return peaks_in_hist, peaks, hist_no_zero, filtered
    
    def MultiOtsu(self,hist, num_classes):
        if num_classes<2:
            return []
        if num_classes>4: 
            original_distri_x = list(hist[0])
            original_bins_x = list(hist[1])
            # downsample the histogram
            distri_x = original_distri_x[::5]
            bins_x = original_bins_x[::5]
            hist_new = (np.array(distri_x), np.array(bins_x))
            thresholds = threshold_multiotsu(image=None, classes= num_classes, hist=hist_new)
        else:
            thresholds = threshold_multiotsu(image=None, classes= num_classes, hist=hist)
        return thresholds

    def CuttingEdage(self, mask_frame, original_mask, threshold_x, stride = 0.8, lambda_x = 0.6):
        Y_dim, X_dim = mask_frame.shape
        mask_2 = np.ones_like(mask_frame)
        initial_cutting_edges_x = []
        for t in threshold_x:
            t = int(t)
            edge = [t for i in range(Y_dim)]
            initial_cutting_edges_x.append(edge)
        
        updated_cutting_edges_x = []
        for index, e_x in enumerate(initial_cutting_edges_x):
            temp_edge = []
            last_drift = 0
            last_cut_pt = e_x[0]
            for row in range(Y_dim):
                x_left = max(e_x[row]-20, 0)
                x_right = min(e_x[row] + 20, X_dim)
                left_mean = np.mean(mask_frame[row, x_left:e_x[row]])
                right_mean = np.mean(mask_frame[row, e_x[row]:x_right])
                diff = left_mean - right_mean
                drift = lambda_x * diff + (1-lambda_x) * last_drift
                current_cut_pt = int(drift*stride + e_x[row])
                temp_edge.append(current_cut_pt)
                if current_cut_pt < last_cut_pt:
                    mask_2[row, current_cut_pt:last_cut_pt+1] = 0
                else:
                    mask_2[row, last_cut_pt: current_cut_pt+1] = 0
                last_cut_pt = current_cut_pt
                last_drift = drift
            updated_cutting_edges_x.append(temp_edge)
        return mask_2 * original_mask
    
    
    def Forward(self, frame):
        """processing

        Args:
            frame (numpy.array): the output of the BaseProcess

        Returns:
            numpy.array: masks of all steps that contain the region of interests (ROIs)
        """
        expansion_coeff = self.expansion_coefficient*self.expansion_coefficient
        box_size = (int(expansion_coeff * self.valid_region_area_limit)//2)*2 +1 # the box_size must be odd.
        maximum = 255
        mask = self.AdaptiveBinary(frame, maximum, box_size)
        mask2 = self.BorderRemove(mask)
        pixel_limit = expansion_coeff * self.valid_region_area_limit
        mask3 = self.BlobRemove(mask2, pixel_limit)
        
        re_mask = np.zeros_like(mask3)
        masks = self.RegionDivid(mask3)
        for m in masks:
            distri_x, bins_x, mask_frame = self.TemperatureDistributionCalc(frame, m)
            
            peaks_x, _ , _, _ = self.CandidateCount_x(distri_x, 20) 
            
            if len(peaks_x) == 0:
                thresholds_x = []
            else:
                thresholds_x = self.MultiOtsu((distri_x,bins_x), num_classes = len(peaks_x))
            
            mask1 = self.CuttingEdage(mask_frame, m, thresholds_x, stride = 0.4, lambda_x = 0.4)
            mask2 = self.TopKRegion(mask1, topk = (len(thresholds_x) + 1))
            re_mask = re_mask + mask2
        re_mask = np.where(re_mask>0.2, 255, 0)    
        
        img = re_mask.copy()
        img = np.array(img, np.uint8)
        try:
            contours,hierarchy = cv2.findContours(img,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        except:
            contours = []
        bounding_boxes = []
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            bounding_boxes.append((x,y,w,h))
        
        re_mask_colored = self.RegionColored(re_mask)     
        return mask3, re_mask, re_mask_colored, bounding_boxes

# second component: considering the temporal information
class TrackingProcess():
    def __init__(self) -> None:
        self.trackers = []
        self.timers = []
    
    def CreatTracker(self,frame, initBB):
        # print("Before creating traker, No. of tracker: ", len(self.trackers))
        tracker = cv2.TrackerKCF_create()
        tracker.init(frame, initBB)
        self.trackers.append(tracker)
        self.timers.append(0)
        # print("After creating traker, No. of tracker: ", len(self.trackers))
        
    def DeleteTracker(self,tracker_index):
        # print("No. of Tracker: ", len(self.trackers))
        if tracker_index < len(self.trackers):
            del self.trackers[tracker_index]
            del self.timers[tracker_index]
            return 1
        return 0


    def ReplaceTracker(self, tracker_index, frame, initBB):
        # print("Replacing")
        if tracker_index < len(self.trackers):
            tracker = cv2.TrackerKCF_create()
            tracker.init(frame, initBB)
            self.trackers[tracker_index] = tracker
            self.timers[tracker_index] += 1
            # print("OK")
        else:
            # print("fail")
            return 0
        
    def GetTrackersNum(self):
        return len(self.trackers)
    
    def Forward(self, frame):
        states = []
        timers = []
        return_bboxs = []
        for index,tracker in enumerate(self.trackers):
            state, bbox = tracker.update(frame) 
            self.timers[index] += 1
            states.append(state)
            timers.append(self.timers[index])
            return_bboxs.append(bbox)    
        return states, return_bboxs, timers
        

class TrackingDetectingMergeProcess():
    def __init__(self, expansion_coefficient = 20, valid_region_area_limit = 5, prvs_mask_buffer_size=5) -> None:
        self.detector = DetectingProcess(expansion_coefficient, valid_region_area_limit)
        self.tracker = TrackingProcess()
        self.prvs_mask_buffer_size = prvs_mask_buffer_size
        self.prvs_mask_buffer = []
    
    def OverlappingFilter(self,current_mask):
        overlapped_points = []
        if len(self.prvs_mask_buffer) == 0:
            return current_mask, overlapped_points
        overlapping = current_mask.copy()
        for prvs_mask in self.prvs_mask_buffer:
            overlapping = prvs_mask * overlapping
        current_labeled = label(current_mask)
        output_mask = np.zeros_like(current_mask)
        
        rps = regionprops(current_labeled)
        for i in range(len(rps)):
            corordinates = rps[i].coords.T
            overlapped_flag = False
            for j in range(len(corordinates[0])):
                x = corordinates[0][j]
                y = corordinates[1][j]
                if overlapping[x,y] > 0.01:
                    overlapped_flag = True
                    overlapped_points.append((x,y))
                    break
            if overlapped_flag:
                output_mask[tuple(corordinates)] = 255
        return output_mask, overlapped_points
    
    def FillHoles(self, mask):
        re_mask = ndi.binary_fill_holes(mask)
        return re_mask
    
    def FindBBox(self, mask):
        img = mask.copy()
        img_temp = self.FillHoles(img)
        img = np.array(img_temp, np.uint8)
        try:
            contours,hierarchy = cv2.findContours(img,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        except:
            contours = []
        areas = [] 
        for cnt in contours:
            areas.append(cv2.contourArea(cnt))
        areas_np = np.array(areas)
        sort_index = np.argsort(areas_np)
        inverse_sort_index = sort_index[::-1]
        
        bounding_boxes = []
        re_areas = []
        for index in inverse_sort_index:
            cnt = contours[index]
            x,y,w,h = cv2.boundingRect(cnt)
            bounding_boxes.append((x,y,w,h))
            re_areas.append(areas[index])
        
        return bounding_boxes,re_areas
    
    def RegionColored(self,mask_in):
        # Given different region with different values
        mask = mask_in.copy()
        mask_labeled = label(mask)
        rps = regionprops(mask_labeled)
        num_regions = len(rps)
        if num_regions == 0:
            return mask
        values = np.linspace(70,190,num_regions)
        re_mask = np.zeros_like(mask)
        for i in range(len(rps)):
            re_mask[tuple(rps[i].coords.T)] = values[i]
        return re_mask
    
    def RegionDivid(self, mask):
        masks = []
        copy_mask = mask.copy()
        label_mask = label(copy_mask)
        rps = regionprops(label_mask)
        areas = [r.area for r in rps]
        for i in range(len(rps)):
            corordinates = rps[i].coords.T
            temp_mask = np.zeros_like(mask)
            temp_mask[tuple(corordinates)] = 1
            masks.append(temp_mask)
        return masks, areas
    
    def HorizontalCutting(self, frame, mask, bounds, stride = 0.8, lambda_y = 0.8):
        temp_frame = frame.copy()
        temp_mask = np.where(mask>0.2, 1, 0)
        masked_frame = temp_frame*temp_mask

        Y_dim, X_dim = masked_frame.shape
        cutting_mask = np.ones_like(masked_frame)
        hist = np.max(masked_frame, axis = 1)
        cutting_anchors = []
        for bound in bounds:
            upper_bound, lower_bound = bound
            # print("np.argmin(hist[upper_bound:lower_bound]): ", np.argmin(hist[upper_bound:lower_bound]))
            cutting_anchor = np.argmin(hist[upper_bound:lower_bound])
            # cutting_anchors.append(upper_bound + cutting_anchor)

            cutting_anchors.append(lower_bound)
            # print("cutting_anchors: ", cutting_anchors)
        
        initial_cutting_edges_y = []
        for cutting_anchor in cutting_anchors:
            cutting_mask[cutting_anchor, :] = 0
            edge = [cutting_anchor for i in range(X_dim)]
            initial_cutting_edges_y.append(edge)
        return cutting_mask
    
    def VerticalCutting(self, frame, mask, bounds, stride = 0.8, lambda_x = 0.6):
        temp_frame = frame.copy()
        temp_mask = np.where(mask>0.2, 1, 0)
        masked_frame = temp_frame*temp_mask
        
        Y_dim, X_dim = masked_frame.shape
        mask_2 = np.ones_like(masked_frame)
        initial_cutting_edges_x = []
        for bound in bounds:
            left_bound, right_bound = bound
            t = int((left_bound + right_bound)/2)
            edge = [t for i in range(Y_dim)]
            initial_cutting_edges_x.append(edge)
        
        updated_cutting_edges_x = []
        for index, e_x in enumerate(initial_cutting_edges_x):
            temp_edge = []
            last_drift = 0
            last_cut_pt = e_x[0]
            for row in range(Y_dim):
                x_left = max(e_x[row]-20, 0)
                x_right = min(e_x[row] + 20, X_dim)
                left_mean = np.mean(masked_frame[row, x_left:e_x[row]])
                right_mean = np.mean(masked_frame[row, e_x[row]:x_right])
                diff = left_mean - right_mean
                drift = lambda_x * diff + (1-lambda_x) * last_drift
                current_cut_pt = int(drift*stride + e_x[row])
                temp_edge.append(current_cut_pt)
                if current_cut_pt < last_cut_pt:
                    mask_2[row, current_cut_pt:last_cut_pt+1] = 0
                else:
                    mask_2[row, last_cut_pt: current_cut_pt+1] = 0
                last_cut_pt = current_cut_pt
                last_drift = drift
            updated_cutting_edges_x.append(temp_edge)
        return mask_2
    

    def RegionsInBox(self, mask, BBox):
        (x, y, w, h) = BBox
        InBBox_Area = mask[y:y+h, x:x+w]
        center_pt = InBBox_Area[int(h/2), int(w/2)]
        masks,areas = self.RegionDivid(InBBox_Area)
        return len(masks), center_pt, areas
      
    def Forward(self, frame_gray):
        # Detecting part
        mask, x_split_mask, x_split_mask_colored, _ = self.detector.Forward(frame_gray)
        x_split_mask = np.where(x_split_mask>0.1, 1, 0)

        # Considering the overlapping along the time dimension
        filtered_mask, overlapped_points = self.OverlappingFilter(x_split_mask)
        filtered_mask = np.where(filtered_mask>0.1, 1, 0)
        
        if len(self.prvs_mask_buffer) == 0:
            prvs_mask_colored = self.RegionColored(mask) 
        else:
            prvs_mask_colored = self.RegionColored(self.prvs_mask_buffer[-1])
        
        self.prvs_mask_buffer.append(mask)
        if len(self.prvs_mask_buffer) > self.prvs_mask_buffer_size:
            self.prvs_mask_buffer.pop(0)

        detected_bboxes,_ = self.FindBBox(filtered_mask)
        filtered_mask_colored = self.RegionColored(filtered_mask)
        original_timers = [0 for i in range(len(detected_bboxes))]
        valid_timers = original_timers

        # Tracking part
        frame = cv2.applyColorMap(frame_gray, cv2.COLORMAP_JET)
        if self.tracker.GetTrackersNum() == 0:
            for initBB in detected_bboxes:
                self.tracker.CreatTracker(frame, initBB)
            return mask, x_split_mask_colored, filtered_mask_colored, prvs_mask_colored, detected_bboxes,original_timers, detected_bboxes, valid_timers
        else:
            states, tracking_boxes, timers = self.tracker.Forward(frame)
        
        original_BBoxes = []
        original_timers = []
        invalid_tracking_index = []
        valid_BBoxes = []
        valid_timers = []
        valid_BBoxes_center_pts = []
        valid_box_index2tracker_index = []
        occupied_place = np.zeros_like(filtered_mask)
        center_pts_marker_map = np.zeros_like(filtered_mask)
        H, W = filtered_mask.shape
        for index, state in enumerate(states):
            if state:
                (x, y, w, h) = [int(v) for v in tracking_boxes[index]]
                original_BBoxes.append((x, y, w, h))
                original_timers.append(timers[index])
                regions_in_box, center_pt, areas = self.RegionsInBox(filtered_mask,(x, y, w, h))
                if regions_in_box == 1:
                    if x < 5 and y+h > H-5:
                        invalid_tracking_index.append(index)
                        continue
                    if x+w > W+5 and y+h > H+5:
                        invalid_tracking_index.append(index)
                        continue
                    if center_pt > 0.1:
                        pass
                    else:
                        invalid_tracking_index.append(index)
                        continue
                    occupied_part = occupied_place[y : y + h, x:x+w]
                    overlapped_ratio = np.sum(occupied_part) / (h * w)
                    # if overlapped_ratio > 0.5:
                    #     invalid_tracking_index.append(index)
                    #     continue
                    occupied_place[y:y + h, x:x+w] = 1
                    center_pts_marker_map[int(y + h/2), int(x+w/2)] = 1
                    valid_BBoxes_center_pts.append(( int(y + h/2), int(x+w/2)))
                    valid_BBoxes.append((x, y, w, h))
                    valid_timers.append(timers[index])
                    valid_box_index2tracker_index.append(index)
                elif regions_in_box == 2:
                    a1 = areas[0]
                    a2 = areas[1]
                    if a1 > a2:
                        a_ratio = a2 / a1
                    else:
                        a_ratio = a1 / a2

                    if a_ratio < 0.2:
                        occupied_part = occupied_place[y : y + h, x:x+w]
                        overlapped_ratio = np.sum(occupied_part) / (h * w)
                        # if overlapped_ratio > 0.5:
                        #     invalid_tracking_index.append(index)
                        #     continue
                        occupied_place[y:y + h, x:x+w] = 1
                        center_pts_marker_map[int(y + h/2), int(x+w/2)] = 1
                        valid_BBoxes_center_pts.append(( int(y + h/2), int(x+w/2)))
                        valid_BBoxes.append((x, y, w, h))
                        valid_timers.append(timers[index])
                        valid_box_index2tracker_index.append(index)
                    else:
                        invalid_tracking_index.append(index)
                        continue
                else:
                    invalid_tracking_index.append(index)
            else:
                invalid_tracking_index.append(index)
            
        # self.tracker.DeleteTracker(index)
                
        masks, areas = self.RegionDivid(filtered_mask)
        for i_m,mask_ in enumerate(masks):
            temp_mask = mask_ * center_pts_marker_map
            temp_occupied = mask_ * occupied_place
            x_index, y_index = np.where(temp_mask>0)
            mask_area = areas[i_m]
            # if len(x_index) == 0 and np.sum(temp_occupied) < 20:
            if len(x_index) == 0:
                # there is no tracking box in this region
                detected_bboxes, _ = self.FindBBox(mask_)
                if len(detected_bboxes) == 0:
                    continue
                self.tracker.CreatTracker(frame,detected_bboxes[0])
                valid_BBoxes.append(detected_bboxes[0])
                valid_timers.append(0)
            elif len(x_index) == 1:
                x = x_index[0]
                y = y_index[0]
                detected_bboxes, box_areas = self.FindBBox(mask_)
                if len(detected_bboxes) == 0:
                    continue
                detected_bbox = detected_bboxes[0]
                detected_area = detected_bbox[2] * detected_bbox[3]
                BBox_index = valid_BBoxes_center_pts.index((x,y))
                tracked_bbox = valid_BBoxes[BBox_index]
                tracked_area = tracked_bbox[2] * tracked_bbox[3]

                if detected_area > 1.5*tracked_area:
                    temp_mask = np.zeros_like(mask_)
                    temp_mask[tracked_bbox[1]:tracked_bbox[1]+tracked_bbox[3], tracked_bbox[0]:tracked_bbox[0]+tracked_bbox[2]] = 1
                    cutting_mask = temp_mask * mask_
                    filtered_mask = filtered_mask * cutting_mask
                else:
                    try:
                        valid_BBoxes[BBox_index] = detected_bboxes[0]
                    except:
                        pass
            elif len(x_index) == 2:
                c_x_1 = x_index[0]
                c_y_1 = y_index[0]
                BBox_index_1= valid_BBoxes_center_pts.index((c_x_1,c_y_1))
                Box_1 = valid_BBoxes[BBox_index_1]
                
                c_x_2 = x_index[1]
                c_y_2 = y_index[1]
                BBox_index_2 = valid_BBoxes_center_pts.index((c_x_2,c_y_2))
                Box_2 = valid_BBoxes[BBox_index_2]

                horizontal_overlap_ratio = 0
                vertical_cutting_bounds = []
                if Box_1[0] < Box_2[0]:
                    box_1_right_border = Box_1[0] + Box_1[2]
                    horizontal_overlap_ratio = (box_1_right_border - Box_2[0]) / Box_1[2]
                    if horizontal_overlap_ratio < 0.4:
                        vertical_cutting_bounds.append((box_1_right_border, Box_2[0]))
                else:
                    box_2_right_border = Box_2[0] + Box_2[2]
                    horizontal_overlap_ratio = (box_2_right_border - Box_1[0]) / Box_2[2]
                    if horizontal_overlap_ratio < 0.4:
                        vertical_cutting_bounds.append((box_2_right_border, Box_1[0]))
                
                if len(vertical_cutting_bounds) > 0:
                    cutting_mask = self.VerticalCutting(frame_gray, mask_, vertical_cutting_bounds)
                    filtered_mask = filtered_mask * cutting_mask
                    cutting_result = mask_* cutting_mask
                    cutting_sub_regions = self.RegionDivid(cutting_result)
                    # print("The number of cutting areas (Horizontal): ", len(cutting_sub_regions))
                    detected_bboxes_temp = []
                    detected_bboxes_temp,areas_temp = self.FindBBox(cutting_result)
                    if len(detected_bboxes_temp) < 2:
                        # print("No replacing")
                        continue
                    x_, y_, w_, h_ = detected_bboxes_temp[0]
                    # if c_x_1 > x_ and c_x_1 < x_ + w_:
                    #     tracker_index = valid_box_index2tracker_index[BBox_index_1]
                    #     r1 = self.tracker.ReplaceTracker(tracker_index,frame, detected_bboxes_temp[0])
                    #     print("Replacing:", tracker_index, r1)
                    #     tracker_index = valid_box_index2tracker_index[BBox_index_2]
                    #     r2 = self.tracker.ReplaceTracker(tracker_index,frame, detected_bboxes_temp[1])
                    #     print("Replacing:", tracker_index, r2)
                    # else:
                    #     tracker_index = valid_box_index2tracker_index[BBox_index_1]
                    #     r1 = self.tracker.ReplaceTracker(tracker_index,frame, detected_bboxes_temp[1])
                    #     print("Replacing:", tracker_index, r1)
                    #     tracker_index = valid_box_index2tracker_index[BBox_index_2]
                    #     r2 = self.tracker.ReplaceTracker(tracker_index,frame, detected_bboxes_temp[0])
                    #     print("Replacing:", tracker_index, r2)
                else:
                    horizontal_cutting_bounds = []
                    up_box_index = 0
                    if c_x_1 > c_x_2 and Box_1[1] > Box_2[1]:
                        # valid_BBoxes.append(Box_prvs)
                        # valid_BBoxes.append(Box_current)
                        # print("Boxes:", Box_prvs, Box_current)
                        horizontal_cutting_bounds.append((int(Box_2[1]), int(Box_1[1])))
                        up_box_index = 1
                        # print("cutting:", ((int(Box_prvs[1]),int(Box_current[1]))))
                    elif c_x_1 < c_x_2 and Box_1[1] < Box_2[1]:
                        horizontal_cutting_bounds.append((int(Box_1[1]), int(Box_2[1])))
                    else:
                        tracker_index = valid_box_index2tracker_index[BBox_index_1]
                        invalid_tracking_index.append(tracker_index)
                        # self.tracker.DeleteTracker(BBox_index_prvs)
                    if len(horizontal_cutting_bounds) > 0:
                        cutting_mask = self.HorizontalCutting(frame_gray, mask_, horizontal_cutting_bounds)
                        filtered_mask = filtered_mask * cutting_mask

                        cutting_result = mask_* cutting_mask
                        cutting_sub_regions = self.RegionDivid(cutting_result)
                        # print("The number of cutting areas: ", len(cutting_sub_regions))
                        detected_bboxes_temp = []
                        detected_bboxes_temp,areas_temp = self.FindBBox(cutting_result)
                        if len(detected_bboxes_temp) < 2:
                            # print("No replacing")
                            continue
                        if up_box_index == 0:
                            tracker_index = valid_box_index2tracker_index[BBox_index_1]
                            r1 = self.tracker.ReplaceTracker(tracker_index,frame, detected_bboxes_temp[0])
                            # print("Replacing:", tracker_index, r1)
                            tracker_index = valid_box_index2tracker_index[BBox_index_2]
                            r2 = self.tracker.ReplaceTracker(tracker_index,frame, detected_bboxes_temp[1])
                            # print("Replacing:", tracker_index, r2)
                        else:
                            tracker_index = valid_box_index2tracker_index[BBox_index_1]
                            r1 = self.tracker.ReplaceTracker(tracker_index,frame, detected_bboxes_temp[1])
                            # print("Replacing:", tracker_index, r1)
                            tracker_index = valid_box_index2tracker_index[BBox_index_2]
                            r2 = self.tracker.ReplaceTracker(tracker_index,frame, detected_bboxes_temp[0])
                            # print("Replacing:", tracker_index, r2)
            else:
                for i in range(len(x_index)):
                    x = x_index[i]
                    y = y_index[i]
                    BBox_index = valid_BBoxes_center_pts.index((x,y))
                    tracker_index = valid_box_index2tracker_index[BBox_index]
                    invalid_tracking_index.append(tracker_index)
                    valid_BBoxes[BBox_index] = (0,0,0,0)
        # print("Before // No. tracker: " , len(self.tracker.trackers))
        invalid_tracking_index.sort() 
        # print("invalid_tracking_index:  ", invalid_tracking_index)
        deleted_num = 0
        for invalid_tracker_id in invalid_tracking_index:
            re = self.tracker.DeleteTracker(invalid_tracker_id - deleted_num)
            if re == 1:
                deleted_num += 1
        # print("deleted_num:  ", deleted_num)
        filtered_mask_colored = self.RegionColored(filtered_mask)

        # print("No. tracker: " , len(self.tracker.trackers))
        
        return mask, x_split_mask_colored, filtered_mask_colored, prvs_mask_colored, original_BBoxes, original_timers, valid_BBoxes, valid_timers



class ROIPooling():
    def __init__(self,resize_shape, window_size, stride) -> None:
        self.resize_shape = resize_shape
        self.m = nn.MaxPool2d(window_size,stride)
    
    def PoolingTorch(self, roi):
        resized_roi = cv2.resize(roi, self.resizes_shape)
        roi_tensor = torch.unsqueeze(torch.from_numpy(resized_roi), 0)
        return self.m(roi_tensor)
    
    def PoolingNumpy(self, roi):
        resized_roi = cv2.resize(roi, self.resize_shape)
        roi_tensor = torch.unsqueeze(torch.from_numpy(resized_roi), 0)
        temp = self.m(roi_tensor)
        return temp.cpu().detach().numpy()[0]
    
"""
models:


Kernel ridge regression (KRR): “Machine Learning: A Probabilistic Perspective” Murphy, K. P. - chapter 14.4.3, pp. 492-493, The MIT Press, 2012
krr_rbf: KernelRidge(kernel="rbf", gamma=0.1)
krr_poly: KernelRidge(kernel="poly", gamma="auto", degree=3, coef0=1)
krr_sigmoid: KernelRidge(kernel="sigmoid", gamma="auto", coef0=1)

SVR: LIBSVM: A Library for Support Vector Machines
svr_rbf
svr_poly
svr_sigmoid

DecisionTreeRegressor: dtr
KNN: knn

AdaBoostRegressor: Drucker. “Improving Regressors using Boosting Techniques”, 1997.
abr

"""
class Estimator():
    def __init__(self, model_name = "krr_rbf", ensemble = 1, topk = 1) -> None:
        """
        if ensemble == 1:
            Using topk temperature as input.
        else:
            Using ensembled model, and ignoring topk.
        """
        self.ensemble = ensemble
        self.topk = topk
        # decrease_monotonic_cst = [1 for i in range(topk)]
        self.models = []
        self.trained_models = []
        self.trained_model_scores = []
        for i in range(ensemble):
            if model_name == "krr_rbf":
                self.models.append(KernelRidge(kernel="rbf", gamma=0.1))  
            elif model_name == "svr_rbf":
                self.models.append(SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1))    
            elif model_name == "svr_poly":
                self.models.append(SVR(kernel="poly", C=100, gamma="auto", degree=3, epsilon=0.1, coef0=1))    
            elif model_name == "svr_sigmoid":
                self.models.append(SVR(kernel="sigmoid", C=100, gamma="auto", epsilon=0.1, coef0=1)) 
            elif model_name == "knn":
                self.models.append(neighbors.KNeighborsRegressor(n_neighbors = 5, weights='distance'))
            elif model_name == "dtr":
                self.models.append(DecisionTreeRegressor(max_depth=100))
            elif model_name == "abr":
                self.models.append(AdaBoostRegressor(random_state=0, n_estimators=100))
            elif model_name == "hgbr":
                self.models.append(HistGradientBoostingRegressor())
            elif model_name == "gbr":
                self.models.append(GradientBoostingRegressor(random_state=0, n_estimators = 500 ))
            else:
                self.models.append(SVR(kernel="poly", C=100, gamma="auto", degree=3, epsilon=0.1, coef0=1)) 
                
    def Training(self, X,Y):
        if self.ensemble > 1:
            if len(self.models) != X.shape[1]:
                print("'ensemble' should equal to the number of features of each sample")
                return 0
            
            for index in range(len(self.models)):
                input = np.squeeze(X[:, index]).reshape(-1,1)
                m = self.models[index].fit(input, Y)
                self.trained_models.append(m)
                score = m.score(input, Y)
                self.trained_model_scores.append(score)
        else:
            if self.topk > X.shape[1]:
                print("'topk should less than the number of features of each sample")
                return 0
            for index in range(len(self.models)):
                m = self.models[index].fit(X[:, 0:self.topk], Y)
                self.trained_models.append(m)
                score = m.score(X[:, 0:self.topk], Y)
                self.trained_model_scores.append(score)
        return self.trained_model_scores
    
    
    def Testing(self, X_test):
        Y_predict = []
        if self.ensemble > 1:
            for index in range(len(self.trained_models)):
                input = np.squeeze(X_test[:, index]).reshape(-1,1)
                pred = self.trained_models[index].predict(input)
                Y_predict.append(pred)
        else:
            for index in range(len(self.trained_models)):
                pred = self.trained_models[index].predict(X_test[:, 0:self.topk])
                Y_predict.append(pred)
        return Y_predict


def SizeBasedDepthPredection(depth_list, size_list, buffer_size):
    # if len(depth_list) != len(size_list):
    #     print("Error: depth_list and size_list should have the same length")
    #     return None
    current_W,current_H = size_list[-1]
    current_depth = depth_list[-1]
    predict_depth = []
    
    if len(depth_list) > buffer_size:
        # from -2 to -buffer_size-1
        for i in range(-2,-buffer_size-1,-1):
            p_depth = depth_list[i]
            p_W,p_H = size_list[i]
            
            W_ratio = current_W/p_W
            H_ratio = current_H/p_H
            ratio = min(W_ratio,H_ratio)
            predict_depth.append(p_depth*ratio)
    else:
        for i in range(len(depth_list)-1):
            try:
                p_depth = depth_list[i]
                p_W,p_H = size_list[i]
                
                W_ratio = current_W/p_W
                H_ratio = current_H/p_H
                ratio = min(W_ratio,H_ratio)
                predict_depth.append(p_depth*ratio) 
            except:
                pass
        predict_depth = predict_depth[::-1] 
    if len(predict_depth) < buffer_size:
        # padding with the current depth to make the length of the prediction buffer equal to buffer_size
        predict_depth = [current_depth]*(buffer_size-len(predict_depth)) + predict_depth
    return predict_depth


def WeightsGenerate(buffersize):
    # Define the parameters for the discrete Gaussian distribution
    mean = 0
    std_dev = 0.5
    num_points = 2*buffersize - 1

    # Define a discrete Gaussian distribution function
    def discrete_gaussian(x, mean, std_dev):
        return (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-((x - mean) ** 2) / (2 * std_dev ** 2))

    # Create an array of equally spaced integers
    x_values = np.linspace(mean - 3 * std_dev, mean + 3 * std_dev, num_points)
    # Calculate the corresponding discrete Gaussian probabilities
    y_values = discrete_gaussian(x_values, mean, std_dev)

    weights = y_values[buffersize-1:]
    # normalize the weights
    weights = weights / weights.sum()
    # print(weights)
    # print(weights.sum())
    
    return weights

def discard_outliers_and_find_expectation(data, threshold=2):
    # Calculate the mean and standard deviation of the dataset
    mean = np.mean(data)
    std_dev = np.std(data)

    # Determine the acceptable range for outliers
    lower_bound = mean - threshold * std_dev
    upper_bound = mean + threshold * std_dev

    # Discard data points that fall outside of the acceptable range
    data_filtered = [x for x in data if lower_bound <= x <= upper_bound]
    
    # generate weights for the remaining data points
    buffersize = len(data_filtered)
    weights = WeightsGenerate(buffersize)
    
    # Calculate the new mean (expectation) of the remaining data points under the weights distribution
    new_mean = np.sum(data_filtered*weights)

    # Return the new mean and the filtered data
    return new_mean, data_filtered


def sliding_window_average(data, window_size):
    data = np.array(data)
    data = np.convolve(data, np.ones((window_size,))/window_size, mode='valid')
    return data

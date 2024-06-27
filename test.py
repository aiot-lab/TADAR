import time
import sys
import pickle
import numpy as np
import os
import random
import cv2
import pandas as pd
import datetime
from functions2 import *
from dataset import Dataset
from metrics import ROIDetectionEvaluation,DetectionMeasurements, AverageRelativeError, RMSE, MAE, empirical_cdf, MAEAtEachSection
from tqdm import tqdm
from tsmoothie.smoother import KalmanSmoother
import matplotlib.pyplot as plt
from metrics import *
    

def test(testdata_path, depth_model = None, range_model = None,range_model2=None):
    """the testing function for the detector and range/depth estimator

    Args:
        testdata_path (_type_): A list of the test file names where the files are in Dataset folder
        depth_model (_type_, optional): the saved trained depth model path (all the models are in the Models/ folder). Defaults to None.
        range_model (_type_, optional): the saved trained range model path (all the models are in the Models/ folder). Defaults to None.
        range_model2 (_type_, optional): the saved trained range2 (using the same input of the depth model while output the estimated range) model path (all the models are in the Models/ folder). Defaults to None.

    Returns:
        dictionary: the results of the detector and the estimators.
    """
    # detector configuration
    expansion_coefficient = 20
    temperature_upper_bound = 37
    valid_region_area_limit = 5
    ROIevaluationThreshold = 0.5
    prepipeline = PrePipeline(expansion_coefficient, temperature_upper_bound)
    detector = TrackingDetectingMergeProcess(expansion_coefficient, valid_region_area_limit)

    # estimator configuration
    Roi_Pooling_Size = (2,4)
    resize_shape = (100 * Roi_Pooling_Size[0], 100 * Roi_Pooling_Size[1]) 
    window_size = 100
    roipooling = ROIPooling(resize_shape, window_size,window_size)
    topk = 10
    if topk > Roi_Pooling_Size[0] * Roi_Pooling_Size[1]:
        topk = Roi_Pooling_Size[0] * Roi_Pooling_Size[1]
    topk = Roi_Pooling_Size[0] * Roi_Pooling_Size[1]
    kalman_smoother = KalmanSmoother(component='level_trend', 
                                    component_noise={'level':0.1, 'trend':0.0000001})

    ## dataset loading
    testset = Dataset(testdata_path)

    if range_model is not None:
        range_estimator = pickle.load(open(range_model, 'rb'))
    if range_model2 is not None:
        range_estimator2 = pickle.load(open(range_model2, 'rb'))
    if depth_model is not None:
        depth_estimator = pickle.load(open(depth_model, 'rb'))
    
    TruePositive = []
    FalsePositive = []
    FalseNegtive = []

    Outputs = {
        'ira_matrix': [],  # the input ira matrix
        'frame_index': [], # the corresponding index of the ira and rgb frame
        'GT_range': [], # the ground truth range
        'GT_depth': [],  # the ground truth depth
        'depth_raw_prediction': [], # the raw depth prediction (the output from the depth estimator)
        'depth_KF_smoothed_prediction': [], # the kalman smoothed raw depth prediction
        'depth_Size_based_predictioins': [], # the depth prediction based on the bounding box size change 
        'depth_KF_smoothed_Size_based_predictioins': [], # the depth estimation based on the depth_KF_smoothed_prediction and depth_Size_based_predictioins
        'range_raw_prediction': [],  # the raw range prediction (the output from the range estimator)
        'range_KF_smoothed_prediction': [], # the kalman smoothed raw range prediction
        'range2_raw_prediction': [], # the raw range2 prediction (the output from the range2 estimator)
        'range2_KF_smoothed_prediction': [], # the kalman smoothed raw range2 prediction
    }

    buffer_size = 10

    buffer_pred_range = {
            '0':[],
        }
    buffer_pred_range2 = {
            '0':[],
        }
    buffer_pred_depth = {
            '0':[],
        }
    buffer_pred_WH = {
            '0':[],
        }
    buffer_pred_depth_final = {
            '0':[],
        }

    # for sample_index in tqdm(range(testset.len())):
    for sample_index in range(testset.len()):
        ira_matrix, ambient_temperature, timestamps, GT_bbox, GT_depth, GT_range = testset.GetSample(sample_index)
        ira_img, subpage_type, ira_mat = prepipeline.Forward(ira_matrix, ambient_temperature)
        if not isinstance(ira_img, (np.ndarray)):
            continue
        mask, x_split_mask_colored, filtered_mask_colored, prvs_mask_colored, original_BBoxes,original_timers, valid_BBoxes, valid_timers =  detector.Forward(ira_img)
        result, matched_bbox = ROIDetectionEvaluation(GT_bbox, valid_BBoxes,valid_timers, threshold=ROIevaluationThreshold)
        
        for i_box,ele in enumerate(matched_bbox):
            box, pred_box,count, id, index, IoU = ele
            x,y,w,h = pred_box
            if w == 0:
                continue
            range_ = GT_range[index]
            depth = GT_depth[index]
            temp_roi = ira_mat[int(y):int(y+h), int(x):int(x+w)]
            try:
                pooled_roi = roipooling.PoolingNumpy(temp_roi)
            except:
                continue
            flat_data = np.reshape(np.array([pooled_roi]), (1, -1))
            sort_flat_data = np.sort(flat_data, axis=-1)[:,::-1]
            center_pont = (x+w/2,y+h/2)
            center_data = np.reshape(np.array([center_pont]), (1, -1))
            depth_estimater_input = np.concatenate((sort_flat_data[:,:topk],center_data),axis=1)
            
            Outputs['frame_index'].append(sample_index)
            Outputs['GT_range'].append(range_)
            # range estimation
            range_final_output = None
            if range_model is not None:
                predict_r = range_estimator.predict(sort_flat_data[:,:topk])
                predict_r = predict_r[0]
                Outputs['range_raw_prediction'].append(predict_r)
                # range estimation postprocessing
                if str(id) in buffer_pred_range.keys():
                    if count == 0:
                        buffer_pred_range[str(id)] = []
                        buffer_pred_range[str(id)].append(predict_r)
                        Outputs['range_KF_smoothed_prediction'].append(predict_r)
                        range_final_output = predict_r
                    else:
                        temp_predict = buffer_pred_range[str(id)] + [predict_r]
                        buffer_pred_range[str(id)].append(predict_r)  
                        # kalman smoother
                        kalman_smoother.smooth(temp_predict)
                        kf_smoothed_pred = kalman_smoother.smooth_data[0]
                        kf_predict = kf_smoothed_pred[-1]
                        Outputs['range_KF_smoothed_prediction'].append(kf_predict)
                        range_final_output = kf_predict
                else:
                    buffer_pred_range[str(id)] = []
                    buffer_pred_range[str(id)].append(predict_r)
                    Outputs['range_KF_smoothed_prediction'].append(predict_r)
                    range_final_output = predict_r
            
            
            # range2 estimation, testing the second range estimator which use the same input of the depth estimator while output the estimated range
            range2_final_output = None
            if range_model2 is not None:
                predict_r = range_estimator2.predict(depth_estimater_input)
                predict_r = predict_r[0]
                Outputs['range2_raw_prediction'].append(predict_r)
                # range estimation postprocessing
                if str(id) in buffer_pred_range2.keys():
                    if count == 0:
                        buffer_pred_range2[str(id)] = []
                        buffer_pred_range2[str(id)].append(predict_r)
                        Outputs['range2_KF_smoothed_prediction'].append(predict_r)
                        range2_final_output = predict_r
                    else:
                        temp_predict = buffer_pred_range2[str(id)] + [predict_r]
                        buffer_pred_range2[str(id)].append(predict_r)  
                        # kalman smoother
                        kalman_smoother.smooth(temp_predict)
                        kf_smoothed_pred = kalman_smoother.smooth_data[0]
                        kf_predict = kf_smoothed_pred[-1]
                        Outputs['range2_KF_smoothed_prediction'].append(kf_predict)
                        range2_final_output = kf_predict
                else:
                    buffer_pred_range2[str(id)] = []
                    buffer_pred_range2[str(id)].append(predict_r)
                    Outputs['range2_KF_smoothed_prediction'].append(predict_r)
                    range2_final_output = predict_r
            
                    
            # depth estimation
            depth_final_output = None
            if depth_model is not None:
                predict_d = depth_estimator.predict(depth_estimater_input)
                predict_d = predict_d[0]
                Outputs['GT_depth'].append(depth)
                Outputs['depth_raw_prediction'].append(predict_d)
                # depth estimation postprocessing
                depth_final_output = predict_d
                if str(id) in buffer_pred_depth.keys():
                    if count == 0:
                        buffer_pred_depth[str(id)] = []
                        buffer_pred_WH[str(id)] = []
                        buffer_pred_depth[str(id)].append(predict_d)
                        buffer_pred_WH[str(id)].append([w,h])
                        Outputs['depth_KF_smoothed_prediction'].append(predict_d)
                        Outputs['depth_Size_based_predictioins'].append([predict_d for i in range(buffer_size)])
                        depth_final_output = predict_d
                        buffer_pred_depth_final[str(id)] = []
                        buffer_pred_depth_final[str(id)].append(depth_final_output)
                        Outputs['depth_KF_smoothed_Size_based_predictioins'].append(depth_final_output)                        
                    else:
                        temp_predict = buffer_pred_depth[str(id)] + [predict_d]
                        buffer_pred_depth[str(id)].append(predict_d)
                        buffer_pred_WH[str(id)].append([w,h])
                        
                        kalman_smoother.smooth(temp_predict)
                        kf_smoothed_pred = kalman_smoother.smooth_data[0]
                        kf_predict = kf_smoothed_pred[-1]
                        
                        Outputs['depth_KF_smoothed_prediction'].append(kf_predict)
                        size_based_predictions = SizeBasedDepthPredection(kf_smoothed_pred, buffer_pred_WH[str(id)], buffer_size)
                        Outputs['depth_Size_based_predictioins'].append(size_based_predictions)
                        # add kf_predict to the list: size_based_predictions at the start
                        temp_predict_list = [kf_predict] + size_based_predictions
                        depth_final_output,_ = discard_outliers_and_find_expectation(np.array(temp_predict_list))
                        Outputs['depth_KF_smoothed_Size_based_predictioins'].append(depth_final_output)
                else:
                    buffer_pred_depth[str(id)] = []
                    buffer_pred_WH[str(id)] = []
                    buffer_pred_depth[str(id)].append(predict_d)
                    buffer_pred_WH[str(id)].append([w,h])
                    Outputs['depth_KF_smoothed_prediction'].append(predict_d)
                    Outputs['depth_Size_based_predictioins'].append([predict_d for i in range(buffer_size)])
                    depth_final_output = predict_d
                    buffer_pred_depth_final[str(id)] = []
                    buffer_pred_depth_final[str(id)].append(depth_final_output)
                    Outputs['depth_KF_smoothed_Size_based_predictioins'].append(depth_final_output)
        
        TP, FP, FN =  result
        TruePositive.append(TP)
        FalsePositive.append(FP)
        FalseNegtive.append(FN)
        Outputs['ira_matrix'].append(ira_matrix)

    Outputs['TruePositive'] = TruePositive
    Outputs['FalsePositive'] = FalsePositive
    Outputs['FalseNegtive'] = FalseNegtive

    return Outputs

        
if __name__ == "__main__":
    test_file_pathes = [
        'Dataset/Bathroom1_0_sensor_1.pickle'
        'Dataset/Bathroom1_0_sensor_4.pickle'
        'Dataset/Bathroom1_1_sensor_1.pickle'
        'Dataset/Bathroom1_1_sensor_4.pickle'
        'Dataset/Bedroom1_11_sensor_4.pickle'
        'Dataset/Bedroom1_12_sensor_4.pickle'
        'Dataset/Bedroom1_13_sensor_4.pickle'
        'Dataset/Bedroom1_14_sensor_4.pickle'
        'Dataset/Bedroom1_14_sensor_4.pickle'
        'Dataset/Corridor2_3_sensor_1.pickle'
        'Dataset/Corridor2_3_sensor_4.pickle'
        'Dataset/Corridor2_4_sensor_1.pickle'
        'Dataset/Corridor2_4_sensor_4.pickle'
        'Dataset/Corridor2_5_sensor_1.pickle'
        'Dataset/Corridor2_5_sensor_4.pickle'
        'Dataset/Corridor3_0_sensor_1.pickle'
        'Dataset/Corridor3_0_sensor_4.pickle'
        'Dataset/Corridor3_1_sensor_1.pickle'
        'Dataset/Corridor3_1_sensor_4.pickle'
        'Dataset/Corridor3_2_sensor_1.pickle'
        'Dataset/Corridor3_2_sensor_4.pickle'
        'Dataset/Corridor3_3_sensor_1.pickle'
        'Dataset/Corridor3_3_sensor_4.pickle'
        'Dataset/Corridor3_4_sensor_1.pickle'
        'Dataset/Corridor3_4_sensor_4.pickle'
        'Dataset/Corridor3_5_sensor_1.pickle'
        'Dataset/Corridor3_5_sensor_4.pickle'
        'Dataset/HW101_AmbientObjects_AC_0_sensor_1.pickle'
        'Dataset/HW101_AmbientObjects_AC_0_sensor_4.pickle'
        'Dataset/HW101_AmbientObjects_AC_1_sensor_1.pickle'
        'Dataset/HW101_AmbientObjects_AC_1_sensor_4.pickle'
        'Dataset/HW101_AmbientObjects_display_0_sensor_1.pickle'
        'Dataset/HW101_AmbientObjects_display_0_sensor_4.pickle'
        'Dataset/HW101_AmbientObjects_display_1_sensor_1.pickle'
        'Dataset/HW101_AmbientObjects_display_1_sensor_4.pickle'
        'Dataset/HW101_AmbientObjects_display_2_sensor_1.pickle'
        'Dataset/HW101_AmbientObjects_display_2_sensor_4.pickle'
        'Dataset/HW101_AmbientObjects_hotwaterpot_0_sensor_1.pickle'
        'Dataset/HW101_AmbientObjects_hotwaterpot_0_sensor_4.pickle'
        'Dataset/HW101_AmbientObjects_hotwaterpot_1_sensor_1.pickle'
        'Dataset/HW101_AmbientObjects_hotwaterpot_1_sensor_4.pickle'
        'Dataset/HW101_AmbientObjects_hotwaterpot_2_sensor_1.pickle'
        'Dataset/HW101_AmbientObjects_hotwaterpot_2_sensor_4.pickle'
        'Dataset/HW101_AmbientObjects_laptop_0_sensor_4.pickle'
        'Dataset/HW101_AmbientObjects_laptop_1_sensor_4.pickle'
        'Dataset/HW101_AmbientObjects_laptop_2_sensor_4.pickle'
        'Dataset/HW101_AmbientObjects_router_0_sensor_4.pickle'
        'Dataset/HW101_AmbientObjects_router_1_sensor_4.pickle'
        'Dataset/HW101_AmbientObjects_router_2_sensor_4.pickle'
        'Dataset/HW101_ClothesCoat_0_sensor_1.pickle'
        'Dataset/HW101_ClothesCoat_1_sensor_1.pickle'
        'Dataset/HW101_ClothesCoat_2_sensor_1.pickle'
        'Dataset/HW101_ClothesJacket_0_sensor_1.pickle'
        'Dataset/HW101_ClothesJacket_1_sensor_1.pickle'
        'Dataset/HW101_ClothesJacket_2_sensor_1.pickle'
        'Dataset/HW101_ClothesShirt_0_sensor_1.pickle'
        'Dataset/HW101_ClothesShirt_1_sensor_1.pickle'
        'Dataset/HW101_ClothesShirt_2_sensor_1.pickle'
        'Dataset/HW101_ClothesTshirt_0_sensor_1.pickle'
        'Dataset/HW101_ClothesTshirt_1_sensor_1.pickle'
        'Dataset/HW101_ClothesTshirt_1_sensor_1.pickle'
        'Dataset/HW101_ClothesTshirt_2_sensor_1.pickle'
        'Dataset/HW101_Inangle_15D_0_sensor_1.pickle'
        'Dataset/HW101_Inangle_15D_1_sensor_1.pickle'
        'Dataset/HW101_Inangle_15D_2_sensor_1.pickle'
        'Dataset/HW101_Inangle_15D_2_sensor_1.pickle'
        'Dataset/HW101_Inangle_30D_0_sensor_1.pickle'
        'Dataset/HW101_Inangle_30D_1_sensor_1.pickle'
        'Dataset/HW101_Inangle_30D_2_sensor_1.pickle'
        'Dataset/HW101_Inangle_m15D_1_sensor_1.pickle'
        'Dataset/HW101_Inangle_m15D_3_sensor_1.pickle'
        'Dataset/HW101_Inangle_m15D_3_sensor_4.pickle'
        'Dataset/HW101_Inangle_m30D_0_sensor_1.pickle'
        'Dataset/HW101_Inangle_m30D_1_sensor_1.pickle'
        'Dataset/HW101_Inangle_m30D_2_sensor_1.pickle'
        'Dataset/HW101_Inangle_m30D_3_sensor_1.pickle'
        'Dataset/HW101_Inangle_m30D_3_sensor_4.pickle'
        'Dataset/HW101_Inangle_m30D_4_sensor_1.pickle'
        'Dataset/HW101_Inangle_m30D_4_sensor_4.pickle'
        'Dataset/HW101_Inangle_m30D_5_sensor_1.pickle'
        'Dataset/HW101_Inangle_m30D_5_sensor_4.pickle'
        'Dataset/HW101_LightCond0_0_sensor_1.pickle'
        'Dataset/HW101_LightCond0_1_sensor_1.pickle'
        'Dataset/HW101_LightCond0_2_sensor_1.pickle'
        'Dataset/HW101_LightCond1_0_sensor_1.pickle'
        'Dataset/HW101_LightCond1_1_sensor_1.pickle'
        'Dataset/HW101_LightCond1_2_sensor_1.pickle'
        'Dataset/HW101_LightCond2_0_sensor_1.pickle'
        'Dataset/HW101_LightCond2_1_sensor_1.pickle'
        'Dataset/HW101_LightCond2_2_sensor_1.pickle'
        'Dataset/HW101_LightCond3_0_sensor_1.pickle'
        'Dataset/HW101_LightCond3_1_sensor_1.pickle'
        'Dataset/HW101_LightCond3_2_sensor_1.pickle'
        'Dataset/HW101_Ori_0D_0_sensor_1.pickle'
        'Dataset/HW101_Ori_0D_1_sensor_1.pickle'
        'Dataset/HW101_Ori_0D_2_sensor_1.pickle'
        'Dataset/HW101_Ori_0D_3_sensor_4.pickle'
        'Dataset/HW101_Ori_0D_4_sensor_4.pickle'
        'Dataset/HW101_Ori_0D_5_sensor_4.pickle'
        'Dataset/HW101_Ori_135D_0_sensor_1.pickle'
        'Dataset/HW101_Ori_135D_1_sensor_1.pickle'
        'Dataset/HW101_Ori_135D_2_sensor_1.pickle'
        'Dataset/HW101_Ori_135D_3_sensor_4.pickle'
        'Dataset/HW101_Ori_135D_4_sensor_4.pickle'
        'Dataset/HW101_Ori_135D_5_sensor_4.pickle'
        'Dataset/HW101_Ori_180D_0_sensor_1.pickle'
        'Dataset/HW101_Ori_180D_1_sensor_1.pickle'
        'Dataset/HW101_Ori_180D_2_sensor_1.pickle'
        'Dataset/HW101_Ori_180D_3_sensor_4.pickle'
        'Dataset/HW101_Ori_180D_4_sensor_4.pickle'
        'Dataset/HW101_Ori_180D_5_sensor_4.pickle'
        'Dataset/HW101_Ori_45D_0_sensor_1.pickle'
        'Dataset/HW101_Ori_45D_1_sensor_1.pickle'
        'Dataset/HW101_Ori_45D_2_sensor_1.pickle'
        'Dataset/HW101_Ori_45D_3_sensor_4.pickle'
        'Dataset/HW101_Ori_45D_4_sensor_4.pickle'
        'Dataset/HW101_Ori_45D_5_sensor_4.pickle'
        'Dataset/HW101_Ori_90D_0_sensor_1.pickle'
        'Dataset/HW101_Ori_90D_1_sensor_1.pickle'
        'Dataset/HW101_Ori_90D_2_sensor_1.pickle'
        'Dataset/HW101_Ori_90D_3_sensor_4.pickle'
        'Dataset/HW101_Ori_90D_4_sensor_4.pickle'
        'Dataset/HW101_Ori_90D_5_sensor_4.pickle'
        'Dataset/HW101_W_phone2_sensor_1.pickle'
        'Dataset/HW101_W_phone2_sensor_4.pickle'
        'Dataset/HW101_W_phone3_sensor_4.pickle'
        'Dataset/HW101set10_sensor_1.pickle'
        'Dataset/HW101set11_sensor_1.pickle'
        'Dataset/HW101set12_sensor_1.pickle'
        'Dataset/HW101set13_sensor_1.pickle'
        'Dataset/HW101set14_sensor_1.pickle'
        'Dataset/HW101set15_sensor_1.pickle'
        'Dataset/HW101set16_sensor_1.pickle'
        'Dataset/HW101set17_sensor_1.pickle'
        'Dataset/HW101set18_sensor_1.pickle'
        'Dataset/HW101set19_sensor_1.pickle'
        'Dataset/HW101set1_sensor_1.pickle'
        'Dataset/HW101set20_sensor_1.pickle'
        'Dataset/HW101set21_sensor_1.pickle'
        'Dataset/HW101set22_sensor_1.pickle'
        'Dataset/HW101set23_sensor_1.pickle'
        'Dataset/HW101set24_sensor_1.pickle'
        'Dataset/HW101set24_sensor_1.pickle'
        'Dataset/HW101set25_sensor_1.pickle'
        'Dataset/HW101set26_sensor_1.pickle'
        'Dataset/HW101set27_sensor_1.pickle'
        'Dataset/HW101set28_sensor_1.pickle'
        'Dataset/HW101set29_sensor_1.pickle'
        'Dataset/HW101set2_sensor_1.pickle'
        'Dataset/HW101set30_sensor_1.pickle'
        'Dataset/HW101set30_sensor_1.pickle'
        'Dataset/HW101set31_sensor_1.pickle'
        'Dataset/HW101set32_sensor_1.pickle'
        'Dataset/HW101set33_sensor_1.pickle'
        'Dataset/HW101set34_sensor_1.pickle'
        'Dataset/HW101set34_sensor_1.pickle'
        'Dataset/HW101set35_sensor_1.pickle'
        'Dataset/HW101set36_sensor_1.pickle'
        'Dataset/HW101set37_sensor_1.pickle'
        'Dataset/HW101set38_sensor_1.pickle'
        'Dataset/HW101set39_sensor_1.pickle'
        'Dataset/HW101set3_sensor_1.pickle'
        'Dataset/HW101set40_sensor_1.pickle'
        'Dataset/HW101set41_sensor_1.pickle'
        'Dataset/HW101set42_sensor_1.pickle'
        'Dataset/HW101set43_sensor_1.pickle'
        'Dataset/HW101set44_sensor_1.pickle'
        'Dataset/HW101set45_sensor_1.pickle'
        'Dataset/HW101set46_sensor_1.pickle'
        'Dataset/HW101set47_sensor_1.pickle'
        'Dataset/HW101set48_sensor_1.pickle'
        'Dataset/HW101set49_sensor_1.pickle'
        'Dataset/HW101set4_sensor_1.pickle'
        'Dataset/HW101set50_sensor_1.pickle'
        'Dataset/HW101set51_sensor_1.pickle'
        'Dataset/HW101set52_sensor_1.pickle'
        'Dataset/HW101set53_sensor_1.pickle'
        'Dataset/HW101set54_sensor_1.pickle'
        'Dataset/HW101set55_sensor_1.pickle'
        'Dataset/HW101set56_sensor_1.pickle'
        'Dataset/HW101set57_sensor_1.pickle'
        'Dataset/HW101set58_sensor_1.pickle'
        'Dataset/HW101set59_sensor_1.pickle'
        'Dataset/HW101set5_sensor_1.pickle'
        'Dataset/HW101set60_sensor_1.pickle'
        'Dataset/HW101set61_sensor_1.pickle'
        'Dataset/HW101set6_sensor_1.pickle'
        'Dataset/HW101set7_sensor_1.pickle'
        'Dataset/HW101set8_sensor_1.pickle'
        'Dataset/HW101set9_sensor_1.pickle'
        'Dataset/Hall_0_sensor_1.pickle'
        'Dataset/Hall_0_sensor_4.pickle'
        'Dataset/Hall_1_sensor_1.pickle'
        'Dataset/Hall_1_sensor_4.pickle'
        'Dataset/Hall_2_sensor_1.pickle'
        'Dataset/Hall_2_sensor_4.pickle'
        'Dataset/Meetingroom_0_sensor_1.pickle'
        'Dataset/Meetingroom_0_sensor_4.pickle'
        'Dataset/Meetingroom_1_sensor_1.pickle'
        'Dataset/Meetingroom_1_sensor_4.pickle'
        'Dataset/Meetingroom_2_sensor_1.pickle'
        'Dataset/Meetingroom_2_sensor_4.pickle'
        'Dataset/Meetingroom_3_sensor_1.pickle'
        'Dataset/Meetingroom_3_sensor_4.pickle'
        'Dataset/Meetingroom_4_sensor_1.pickle'
        'Dataset/Meetingroom_4_sensor_4.pickle'
        'Dataset/Meetingroom_5_sensor_1.pickle'
        'Dataset/Meetingroom_5_sensor_4.pickle'
        'Dataset/Meetingroom_5_sensor_4.pickle'
        'Dataset/FiveUser_Dynamic_0_sensor_4.pickle', 
        'Dataset/FiveUser_Dynamic_1_sensor_4.pickle',
        'Dataset/FiveUser_Dynamic_2_sensor_4.pickle',
        'Dataset/FiveUser_Dynamic_3_sensor_4.pickle',
        'Dataset/FiveUser_Dynamic_4_sensor_4.pickle',
        'Dataset/FiveUser_Dynamic_5_sensor_4.pickle',
        'Dataset/FiveUser_Static_0_sensor_4.pickle',
        'Dataset/FiveUser_Static_1_sensor_4.pickle',
        'Dataset/FiveUser_Static_2_sensor_4.pickle',
        'Dataset/FiveUser_Static_3_sensor_4.pickle',
        'Dataset/FiveUser_Static_4_sensor_4.pickle',
        'Dataset/FiveUser_Static_5_sensor_4.pickle',
        'Dataset/FourUser_Dynamic_0_sensor_4.pickle',
        'Dataset/FourUser_Dynamic_1_sensor_4.pickle',
        'Dataset/FourUser_Dynamic_2_sensor_4.pickle',
        'Dataset/FourUser_Dynamic_3_sensor_4.pickle',
        'Dataset/FourUser_Dynamic_4_sensor_4.pickle',
        'Dataset/FourUser_Static_0_sensor_4.pickle',
        'Dataset/FourUser_Static_1_sensor_4.pickle',
        'Dataset/FourUser_Static_2_sensor_4.pickle',
        'Dataset/FourUser_Static_3_sensor_4.pickle',
        'Dataset/FourUser_Static_4_sensor_4.pickle',
    ]

    Results_save_path = 'Outputs/'
    if not os.path.exists(Results_save_path):
        os.makedirs(Results_save_path)

    depth_model = 'Models/hgbr_depth.sav'
    range_model = 'Models/hgbr_range.sav'
    range_model2 = 'Models/hgbr_range2.sav'
    
    if not os.path.exists(depth_model):
        print('ranging model not found')

    failed_file = []
    for file_name in test_file_pathes:
        try:
            print("Starting process: ",file_name)
            sys.stdout.flush()
            testdata_path = [file_name,]
            output = test(testdata_path, depth_model = depth_model, range_model = range_model, range_model2 =range_model2)
            output_filename = file_name.split('/')[-1].split('.')[0]
            with open(Results_save_path + output_filename + '.pkl', 'wb') as f:
                pickle.dump(output, f)
                print('output saved') 
            
            ira_matrix = output['ira_matrix']
            frame_index = output['frame_index']
            GT_range = output['GT_range']
            GT_depth = output['GT_depth']
            depth_raw_prediction = output['depth_raw_prediction']
            depth_KF_smoothed_prediction = output['depth_KF_smoothed_prediction']
            depth_Size_based_predictioins = output['depth_Size_based_predictioins']
            depth_KF_smoothed_Size_based_predictioins = output['depth_KF_smoothed_Size_based_predictioins']
            range_raw_prediction = output['range_raw_prediction']
            range_KF_smoothed_prediction = output['range_KF_smoothed_prediction']
            range2_raw_prediction = output['range2_raw_prediction']
            range2_KF_smoothed_prediction = output['range2_KF_smoothed_prediction']
            TruePositive = output['TruePositive']
            FalsePositive = output['FalsePositive']
            FalseNegtive = output['FalseNegtive']

            ira_matrix = np.array(ira_matrix)
            frame_index = np.array(frame_index)
            GT_range = np.array(GT_range)
            GT_depth = np.array(GT_depth)
            depth_raw_prediction = np.array(depth_raw_prediction)
            depth_KF_smoothed_prediction = np.array(depth_KF_smoothed_prediction)
            depth_Size_based_predictioins = np.array(depth_Size_based_predictioins)
            depth_KF_smoothed_Size_based_predictioins = np.array(depth_KF_smoothed_Size_based_predictioins)
            range_raw_prediction = np.array(range_raw_prediction)
            range_KF_smoothed_prediction = np.array(range_KF_smoothed_prediction)
            range2_raw_prediction = np.array(range2_raw_prediction)
            range2_KF_smoothed_prediction = np.array(range2_KF_smoothed_prediction)

            print("ROI detection Performance: ")
            TP = np.sum(np.array(TruePositive))
            FP = np.sum(np.array(FalsePositive))
            FN = np.sum(np.array(FalseNegtive))
            precision, recall, F1_score = DetectionMeasurements(TP, FP, FN)

            print("precision: ",precision )
            print("recall: ",recall )
            print("F1_score: ",F1_score )

            print('range estimation error:')
            range_prediction = range_KF_smoothed_prediction
            GT = np.array(GT_range)
            range_are, range_are_std = AverageRelativeError(range_prediction ,GT)
            range_rmse , range_rmse_std = RMSE(range_prediction ,GT)
            range_mae, range_mae_std = MAE(range_prediction ,GT)
            range_unique_errors, range_cdf_values = empirical_cdf(range_prediction ,GT)
            range_mesSection, range_mesSection_std = MAEAtEachSection(range_prediction ,GT)
            print('[range_are, std]', range_are, range_are_std)
            print('[range_rmse, std]', range_rmse, range_rmse_std)
            print('[range_mae, std]', range_mae, range_mae_std)
            print('[range_mesSection, std]', range_mesSection, range_mesSection_std)
            
            print('range2 estimation error:')
            range_prediction = range2_KF_smoothed_prediction
            GT = np.array(GT_range)
            range_are, range_are_std = AverageRelativeError(range_prediction ,GT)
            range_rmse , range_rmse_std = RMSE(range_prediction ,GT)
            range_mae, range_mae_std = MAE(range_prediction ,GT)
            range2_unique_errors, range2_cdf_values = empirical_cdf(range_prediction ,GT)
            range2_mesSection, range2_mesSection_std = MAEAtEachSection(range_prediction ,GT)
            print('[range_are, std]', range_are, range_are_std)
            print('[range_rmse, std]', range_rmse, range_rmse_std)
            print('[range_mae, std]', range_mae, range_mae_std)
            print('[range_mesSection, std]', range2_mesSection, range2_mesSection_std)

            print('depth estimation error:')
            GT = np.array(GT_depth)
            print('depth_KF_smoothed_prediction evaluation: ')
            depth_prediction = depth_KF_smoothed_prediction
            depth_are, depth_are_std = AverageRelativeError(depth_prediction ,GT)
            depth_rmse , depth_rmse_std = RMSE(depth_prediction ,GT)
            depth_mae, depth_mae_std = MAE(depth_prediction ,GT)
            depth_unique_errors1, depth_cdf_values1 = empirical_cdf(depth_prediction ,GT)
            depth_mesSection, depth_mesSection_std = MAEAtEachSection(depth_prediction ,GT)
            print('[depth_are, std]', depth_are, depth_are_std)
            print('[depth_rmse, std]', depth_rmse, depth_rmse_std)
            print('[depth_mae, std]', depth_mae, depth_mae_std)
            print('[depth_mesSection, std]', depth_mesSection, depth_mesSection_std)

            print('depth_KF_smoothed_Size_based_predictioins evaluation: ')
            depth_prediction = depth_KF_smoothed_Size_based_predictioins
            depth_are, depth_are_std = AverageRelativeError(depth_prediction ,GT)
            depth_rmse , depth_rmse_std = RMSE(depth_prediction ,GT)
            depth_mae, depth_mae_std = MAE(depth_prediction ,GT)
            depth_unique_errors2, depth_cdf_values2 = empirical_cdf(depth_prediction ,GT)
            depth_mesSection2, depth_mesSection_std2 = MAEAtEachSection(depth_prediction ,GT)
            print('[depth_are, std]', depth_are, depth_are_std)
            print('[depth_rmse, std]', depth_rmse, depth_rmse_std)
            print('[depth_mae, std]', depth_mae, depth_mae_std)
            print('[depth_mesSection, std]', depth_mesSection2, depth_mesSection_std2)
            sys.stdout.flush()

        except KeyboardInterrupt:
            print("Keyboard interrupt")
            break
        except:
            failed_file.append(file_name)
            pass
    print('failed_file: ', failed_file)

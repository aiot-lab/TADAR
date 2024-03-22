import pickle
import numpy as np
import os
import cv2
import pandas as pd
from functions2 import *
from dataset import Dataset
from metrics import ROIDetectionEvaluation,DetectionMeasurements, AverageRelativeError, RMSE, MAE, MAEAtEachSection
from tqdm import tqdm

# data configuration
trainset_datapaths = [    
    'Dataset/train101set1_sensor_1.pickle',
    'Dataset/train101set2_sensor_4.pickle',
    'Dataset/train101set3_sensor_1.pickle',
    'Dataset/train101set4_sensor_4.pickle',
    'Dataset/train101set5_sensor_1.pickle',
    'Dataset/train101set6_sensor_4.pickle',
    'Dataset/train101set7_sensor_1.pickle',
    'Dataset/train101set8_sensor_4.pickle',
    'Dataset/train101set9_sensor_1.pickle',
]

validation_datapaths = [
    'Dataset/train101set1_sensor_4.pickle',
    'Dataset/train101set2_sensor_1.pickle',
    'Dataset/train101set3_sensor_4.pickle',
    'Dataset/train101set4_sensor_1.pickle',
    'Dataset/train101set5_sensor_4.pickle',
    'Dataset/train101set6_sensor_1.pickle',
    'Dataset/train101set7_sensor_4.pickle',
    'Dataset/train101set8_sensor_1.pickle',
    'Dataset/train101set9_sensor_4.pickle',
]

TrainROIFile = 'TrainROI.pickle'
ValidationROIFile = 'ValidationROI.pickle'
reuse_train_roi = False  # if True, the detected ROI will be reused only retrain the estimator
reuse_validation_roi = False
# if False, the ROI will be detected again and the ROI file will be overwritten
# decide whether the roi file is existed or not
if reuse_train_roi: # if we want to reuse the ROI file, the ROI file should be existed
    if os.path.exists(TrainROIFile):
        pass
    else:
        reuse_train_roi = False
        print("Training ROI file not found, ROI need to be detected again")
        exit()
if reuse_validation_roi: # if we want to reuse the ROI file, the ROI file should be existed
    if os.path.exists(ValidationROIFile):
        pass
    else:
        reuse_validation_roi = False
        print("Validation ROI file not found, ROI need to be detected again")
        exit()

if not os.path.exists('Models'):
    os.mkdir('Models')

# detector configuration
expansion_coefficient = 20
temperature_upper_bound = 37
valid_region_area_limit = 5
ROIevaluationThreshold = 0.5
prepipeline = PrePipeline(expansion_coefficient, temperature_upper_bound)
detector = TrackingDetectingMergeProcess(expansion_coefficient, valid_region_area_limit)

# estimator configuration
Roi_Pooling_Size = (2,4)
Model_Name = 'hgbr'
resize_shape = (100 * Roi_Pooling_Size[0], 100 * Roi_Pooling_Size[1]) 
window_size = 100
roipooling = ROIPooling(resize_shape, window_size,window_size)
topk = 8
if topk > Roi_Pooling_Size[0] * Roi_Pooling_Size[1]:
    topk = Roi_Pooling_Size[0] * Roi_Pooling_Size[1]
topk = Roi_Pooling_Size[0] * Roi_Pooling_Size[1]


####################################################################################################
######################################### Start Processing #########################################
####################################################################################################

##### Training:
if not reuse_train_roi:
    ###### Training set ROI detection ########
    print("Training set: ROI detection start")
    ## dataset loading
    trainset = Dataset(trainset_datapaths)
    resize_dim = (640, 480)
    
    ## Stage 1: ROI detection
    TruePositive = []
    FalsePositive = []
    FalseNegtive = []
    ROI = [] # region of interest
    ROI_bbox = [] # the bounding box of the ROI
    ROI_frame_index = [] # the index of the frame where the ROI is detected
    ROI_depth_label = []
    ROI_range_label = []

    for index in tqdm(range(trainset.len())):
        ira_matrix, ambient_temperature, timestamps, GT_bbox, GT_depth, GT_range = trainset.GetSample(index)
        ira_img, subpage_type, ira_mat = prepipeline.Forward(ira_matrix, ambient_temperature)
        if not isinstance(ira_img, (np.ndarray)):
            continue
        mask, x_split_mask_colored, filtered_mask_colored, prvs_mask_colored, original_BBoxes,original_timers, valid_BBoxes, valid_timers =  detector.Forward(ira_img)
        result, matched_bbox = ROIDetectionEvaluation(GT_bbox, valid_BBoxes,valid_timers, threshold=ROIevaluationThreshold)
        TP, FP, FN =  result
        TruePositive.append(TP)
        FalsePositive.append(FP)
        FalseNegtive.append(FN)

        for i_box,ele in enumerate(matched_bbox):
            box, pred_box,count,id, index, IoU = ele
            x,y,w,h = pred_box
            if w == 0:
                continue
            range_ = GT_range[index]
            depth = GT_depth[index]
            if range==0 or depth==0: # which means the depth camera is not available
                print("range_ == 0 or depth == 0")
                continue
            temp_roi = ira_mat[int(y):int(y+h), int(x):int(x+w)]
            ROI_bbox.append([int(x), int(y), int(w), int(h)])
            ROI_frame_index.append(index)
            ROI.append(temp_roi)
            ROI_depth_label.append(depth)
            ROI_range_label.append(range_)

    # save the Training ROI
    train_stage1result = {
        'ROI': ROI,
        'ROI_bbox': ROI_bbox,
        'ROI_frame_index': ROI_frame_index,
        'ROI_depth_label': ROI_depth_label,
        'ROI_range_label': ROI_range_label,
    }
    roi_file = open(TrainROIFile, 'wb')
    pickle.dump(train_stage1result, roi_file)
    roi_file.close()
    print("Trainset: ROI saved")
else:
    # load the roi file
    try: 
        roi_file = open(TrainROIFile, 'rb')
        train_stage1result = pickle.load(roi_file)
        roi_file.close()
        ROI = train_stage1result['ROI']
        ROI_bbox = train_stage1result['ROI_bbox']
        ROI_range_label = train_stage1result['ROI_range_label']
        ROI_depth_label = train_stage1result['ROI_depth_label']
    except:
        print("Training set ROI file is not correct")
        exit()
    

# Stage 2: Depth and Range Estimation models training
preprocessed_roi = []
bbox_centerpoints = []
for index,roi in enumerate(ROI):
    preprocessed_roi.append(roipooling.PoolingNumpy(roi))  # roi pooling
    x,y,w,h = ROI_bbox[index]
    bbox_centerpoints.append([x+w/2, y+h/2])
    
sorted_roi = np.stack(preprocessed_roi) # stack
sorted_roi = np.reshape(sorted_roi, (sorted_roi.shape[0], -1)) # flatten
sorted_roi = np.sort(sorted_roi, axis=-1)[:,::-1] # sort
bbox_centerpoints = np.array(bbox_centerpoints) # get center points
range_label = np.array(ROI_range_label) # get label
depth_label = np.array(ROI_depth_label) # get label
# concatenate the centerpoints and the sorted roi
# depth_estimation_input = np.concatenate((sorted_roi, bbox_centerpoints), axis=1)
depth_estimation_input = np.concatenate((sorted_roi[:,:topk],bbox_centerpoints),axis=1)

range_estimator = Estimator(model_name=Model_Name, ensemble=1, topk=topk)
range_model_score = range_estimator.Training(sorted_roi, range_label)
range_model_filename = 'Models/' + Model_Name + '_range.sav'
pickle.dump(range_estimator.models[0], open(range_model_filename, 'wb'))

# using the depth_input while estimating the range
range_estimator2 = Estimator(model_name=Model_Name, ensemble=1, topk=topk+2)
range2_model_score = range_estimator2.Training(depth_estimation_input, range_label)
range2_model_filename = 'Models/' + Model_Name + '_range2.sav'
pickle.dump(range_estimator2.models[0], open(range2_model_filename, 'wb'))

depth_estimator = Estimator(model_name=Model_Name, ensemble=1, topk=topk+2)
depth_model_score = depth_estimator.Training(depth_estimation_input, depth_label)
depth_model_filename ='Models/' + Model_Name + '_depth.sav'
pickle.dump(depth_estimator.models[0], open(depth_model_filename, 'wb'))

########################################################################################
### Validation:
if not reuse_validation_roi:
    ###### Validation set ROI detection ########
    print("Validation set: ROI detection start")
    ## dataset loading
    validationset = Dataset(validation_datapaths)

    ## Stage 1: ROI detection
    TruePositive = []
    FalsePositive = []
    FalseNegtive = []
    ROI = [] # region of interest
    ROI_bbox = [] # the bounding box of the ROI
    ROI_frame_index = [] # the index of the frame where the ROI is detected
    ROI_depth_label = []
    ROI_range_label = []

    for index in tqdm(range(validationset.len())):
        ira_matrix, ambient_temperature, timestamps, GT_bbox, GT_depth, GT_range = validationset.GetSample(index)
        ira_img, subpage_type, ira_mat = prepipeline.Forward(ira_matrix, ambient_temperature)
        if not isinstance(ira_img, (np.ndarray)):
            continue
        mask, x_split_mask_colored, filtered_mask_colored, prvs_mask_colored, original_BBoxes,original_timers, valid_BBoxes, valid_timers =  detector.Forward(ira_img)
        result, matched_bbox = ROIDetectionEvaluation(GT_bbox, valid_BBoxes,valid_timers, threshold=ROIevaluationThreshold)
        TP, FP, FN =  result
        TruePositive.append(TP)
        FalsePositive.append(FP)
        FalseNegtive.append(FN)

        for i_box,ele in enumerate(matched_bbox):
            box, pred_box,count,id, index, IoU = ele
            x,y,w,h = pred_box
            if w == 0:
                continue
            range_ = GT_range[index]
            depth = GT_depth[index]
            if range_ == 0 or depth == 0:
                print("range_ == 0 or depth == 0")
                continue
            temp_roi = ira_mat[int(y):int(y+h), int(x):int(x+w)]
            ROI_bbox.append([int(x), int(y), int(w), int(h)])
            ROI_frame_index.append(index)
            ROI.append(temp_roi)
            ROI_depth_label.append(depth)
            ROI_range_label.append(range_)

    print("Validationset: ROI detection finished")
    TP = np.sum(np.array(TruePositive))
    FP = np.sum(np.array(FalsePositive))
    FN = np.sum(np.array(FalseNegtive))
    precision, recall, F1_score = DetectionMeasurements(TP, FP, FN)
    print("precision: ",precision )
    print("recall: ",recall )
    print("F1_score: ",F1_score )

    # save the Training ROI
    validation_stage1result = {
        'ROI': ROI,
        'ROI_bbox': ROI_bbox,
        'ROI_frame_index': ROI_frame_index,
        'ROI_depth_label': ROI_depth_label,
        'ROI_range_label': ROI_range_label,
    }
    roi_file = open(ValidationROIFile, 'wb')
    pickle.dump(validation_stage1result, roi_file)
    roi_file.close()
    print("Validation set: ROI saved")
else:
    # load the roi file
    try: 
        roi_file = open(ValidationROIFile, 'rb')
        validation_stage1result = pickle.load(roi_file)
        roi_file.close()
        ROI = validation_stage1result['ROI']
        ROI_bbox = validation_stage1result['ROI_bbox']
        ROI_range_label = validation_stage1result['ROI_range_label']
        ROI_depth_label = validation_stage1result['ROI_depth_label']
    except:
        print("Validation dataset ROI file is not correct")
        exit()
    

# Stage 2: Depth and Range Estimation models validation
preprocessed_roi = []
bbox_centerpoints = []
for index,roi in enumerate(ROI):
    preprocessed_roi.append(roipooling.PoolingNumpy(roi))  # roi pooling
    x,y,w,h = ROI_bbox[index]
    bbox_centerpoints.append([x+w/2, y+h/2])
    
sorted_roi = np.stack(preprocessed_roi) # stack
sorted_roi = np.reshape(sorted_roi, (sorted_roi.shape[0], -1)) # flatten
sorted_roi = np.sort(sorted_roi, axis=-1)[:,::-1] # sort
bbox_centerpoints = np.array(bbox_centerpoints) # get center points
range_label = np.array(ROI_range_label) # get label
depth_label = np.array(ROI_depth_label) # get label
# concatenate the centerpoints and the sorted roi
# depth_estimation_input = np.concatenate((sorted_roi, bbox_centerpoints), axis=1)
depth_estimation_input = np.concatenate((sorted_roi[:,:topk],bbox_centerpoints),axis=1)

predict_range = range_estimator.Testing(sorted_roi)
range_prediction = predict_range[0]
print('range estimation error:')
GT = np.array(range_label)
range_are, range_are_std = AverageRelativeError(range_prediction ,GT)
range_rmse , range_rmse_std = RMSE(range_prediction ,GT)
range_mae, range_mae_std = MAE(range_prediction ,GT)
# range_unique_errors, range_cdf_values = empirical_cdf(range_prediction ,GT)
range_mesSection, range_mesSection_std = MAEAtEachSection(range_prediction ,GT)
print('[range_are, std]', range_are, range_are_std)
print('[range_rmse, std]', range_rmse, range_rmse_std)
print('[range_mae, std]', range_mae, range_mae_std)
print('[range_mesSection, std]', range_mesSection, range_mesSection_std)

# predict range with estimator2
predict_range = range_estimator2.Testing(depth_estimation_input)
range_prediction = predict_range[0]
print('range#2 estimation error:')
GT = np.array(range_label)
range_are, range_are_std = AverageRelativeError(range_prediction ,GT)
range_rmse , range_rmse_std = RMSE(range_prediction ,GT)
range_mae, range_mae_std = MAE(range_prediction ,GT)
# range_unique_errors, range_cdf_values = empirical_cdf(range_prediction ,GT)
range_mesSection, range_mesSection_std = MAEAtEachSection(range_prediction ,GT)
print('[range_are, std]', range_are, range_are_std)
print('[range_rmse, std]', range_rmse, range_rmse_std)
print('[range_mae, std]', range_mae, range_mae_std)
print('[range_mesSection, std]', range_mesSection, range_mesSection_std)

predict_depth = depth_estimator.Testing(depth_estimation_input)
depth_prediction = predict_depth[0]
print('depth estimation error:')
GT = np.array(depth_label)
depth_are, depth_are_std = AverageRelativeError(depth_prediction ,GT)
depth_rmse , depth_rmse_std = RMSE(depth_prediction ,GT)
depth_mae, depth_mae_std = MAE(depth_prediction ,GT)
# depth_unique_errors, depth_cdf_values = empirical_cdf(depth_prediction ,GT)
depth_mesSection, depth_mesSection_std = MAEAtEachSection(depth_prediction ,GT)
print('[depth_are, std]', depth_are, depth_are_std)
print('[depth_rmse, std]', depth_rmse, depth_rmse_std)
print('[depth_mae, std]', depth_mae, depth_mae_std)
print('[depth_mesSection, std]', depth_mesSection, depth_mesSection_std)

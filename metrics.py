import numpy as np
import scipy.stats as stats


def ROIDetectionEvaluation(bbox_label, predicted_bbox, predicted_bbox_counter, threshold = 0.4):
    """
    bbox_label: the ground-truth of bounding boxes, (num_boxes, box), x,y,w,h = box
    predited_bbox: the predicted bounding boxes, (num_boxes, box),
    predicted_bbox_counter: the timer thhat record the number of frames that the predicted bbox is detected. (0 means this is new object, otherwise, it is the number of frames that the same object is detected)
    Return:
    (TP, FP, FN)
    [ground-truth-bbox, predicted-bbox, the lasting time of the object, the object's id ,index of the ground-truth-bbox, IoU]
    """
    matched_bbox = []
    missed_bbox = []
    
    for index, box in enumerate(bbox_label):
        mismatched_flag =False
        g_x, g_y, g_w, g_h =  box
        if g_w == 0:
            continue
        for id, pred_box in enumerate(predicted_bbox):
            p_x, p_y, p_w, p_h =  pred_box
            count = predicted_bbox_counter[id]
            if p_w == 0:
                continue
            
            x_max = max([g_x + g_w, p_x + p_w])
            y_max = max([g_y + g_h, p_y + p_h])
            
            x_min = min([g_x, p_x])
            y_min = min([g_y, p_y])
            
            temp_map =  np.zeros((int(y_max), int(x_max)))            
            temp_map[int(g_y):int(g_y+g_h), int(g_x):int(g_x+g_w)] = 1
            temp_map[int(p_y):int(p_y+p_h), int(p_x):int(p_x+p_w)] += 1
            IoU = np.count_nonzero(temp_map > 1) / ((x_max-x_min) * (y_max-y_min))   
            
            if IoU > threshold:
                matched_bbox.append((box, pred_box, count, id, index, IoU))
                mismatched_flag= True
                break
            
            # since the FOV of IRA is larger than the Depth Camera ground truth, the intersection of the width axis is another metric.
            x_max1 = min([g_x + g_w, p_x + p_w])
            x_min1 = max([g_x, p_x])
            intersection = x_max1 - x_min1
            width_intersection_ratio = intersection / p_w
            
            if IoU > 0.1 and width_intersection_ratio > threshold:
                matched_bbox.append((box, pred_box, count, id, index, IoU))
                mismatched_flag= True
                break
        
        if mismatched_flag:
            missed_bbox.append(box)
    
    TP = len(matched_bbox)
    FP = len(predicted_bbox) - TP
    FN = len(bbox_label) - TP
    
    return (TP, FP, FN), matched_bbox

def DetectionMeasurements(TP, FP, FN):
    if (TP + FP) == 0:
        return 0, 0, 0
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1_score = (2 * precision * recall) / (precision + recall)
    
    return precision, recall, F1_score    

# relative error: error/ground_truth
def AverageRelativeError(prediction,ground_truth):
    error = np.array(prediction) - np.array(ground_truth)
    relative_error = error / np.array(ground_truth + 1e-6)     
    abs_relative_error = np.absolute(relative_error)
    # add vairance of the error
    ere_std = np.std(abs_relative_error)
    sum_relative_error = np.sum(abs_relative_error)
    avg_relative_error = sum_relative_error/ len(prediction)
    return avg_relative_error, ere_std

def RMSE(prediction,ground_truth):
    error = np.array(prediction) - np.array(ground_truth)
    square_error = np.square(error)
    sum_error = np.sum(square_error)
    rmse = np.sqrt(sum_error/ len(prediction))
    # add variation of the error
    rmse_std = np.std(square_error)
    return rmse, rmse_std

def MAE(prediction,ground_truth):
    error = np.array(prediction) - np.array(ground_truth)
    abs_error = np.absolute(error)
    sum_error = np.sum(abs_error)
    mae = sum_error/ len(prediction)
    # add variation of the error
    mae_std = np.std(abs_error)
    return mae, mae_std

def empirical_cdf(pred_depth,ground_truth):
    errors = np.array(pred_depth) - np.array(ground_truth)
    errors = np.abs(errors)
    # Sort the errors in ascending order
    sorted_errors = np.sort(errors)
    # Compute the empirical CDF for each of the unique sorted errors
    unique_errors, cdf_values = np.unique(sorted_errors, return_counts=True)
    cdf_values = np.cumsum(cdf_values) / len(errors)

    # Return the unique errors and their corresponding empirical CDF values
    return unique_errors, cdf_values

# calculate the mean absolute error of the predicted depth at each section of the ground-truth depth, i.e., the error of the predicted depth at the ground-truth depth of 0-0.5m, 0.5-1m, ...., 4.5-5m
def MAEAtEachSection(prediction,ground_truth):
    error = np.array(prediction) - np.array(ground_truth)
    abs_error = np.absolute(error)
    section = np.arange(0, 5.5, 0.5)
    mae = []
    mae_var = []
    for i in range(len(section)-1):
        index = np.where((ground_truth >= section[i]) & (ground_truth < section[i+1]))
        if len(index[0]) == 0:
            mae.append(0)
            mae_var.append(0)
        else:
            mae.append(np.mean(abs_error[index]))
            # add the variation of the error
            mae_var.append(np.var(abs_error[index]))
    return mae, mae_var

def MAEwithErrorBand(prediction,ground_truth, section = np.linspace(0, 5.5, 11)):
    error = np.array(prediction) - np.array(ground_truth)
    abs_error = np.absolute(error)
    # section = np.arange(0, 5.5, 0.5)
    mae = []
    mae_var = []
    for i in range(len(section)-1):
        index = np.where((ground_truth >= section[i]) & (ground_truth < section[i+1]))
        if len(index[0]) == 0:
            mae.append(0)
            mae_var.append(0)
        else:
            mae.append(np.mean(abs_error[index]))
            # add the variation of the error
            mae_var.append(np.var(abs_error[index]))
    return mae, mae_var, section

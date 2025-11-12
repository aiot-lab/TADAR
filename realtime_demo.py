import serial
import time
import ast
import numpy as np
import cv2
import pickle
from tsmoothie.smoother import KalmanSmoother
from functions2 import PrePipeline, TrackingDetectingMergeProcess, ROIPooling, SubpageInterpolating, RegionDivid


def initialize_uart(port='/dev/tty.SLAB_USBtoUART', baud_rate=921600):
    ser = serial.Serial(port, baud_rate, timeout=1)
    if not ser.is_open:
        raise RuntimeError(f"Failed to open serial port {port}")
    print(f"Reading data from {port} at {baud_rate} baud")
    return ser


def preprocess_temperature_data(data_str):
    try:
        dict_data = ast.literal_eval(data_str)
        temperature = np.array(dict_data["temperature"]).reshape((24, 32))
        ambient_temp = dict_data["at"]
        return temperature, ambient_temp
    except:
        return None, None


def apply_color_map(matrix, expansion_coefficient, upper_bound, resize_dim):
    norm = ((matrix - np.min(matrix)) / (upper_bound - np.min(matrix))) * 255
    expanded = np.repeat(np.repeat(norm, expansion_coefficient, axis=0), expansion_coefficient, axis=1)
    colored = cv2.applyColorMap(expanded.astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.resize(colored, resize_dim)


def smooth_predictions(buffer, smoother, predict, max_len=10):
    if len(buffer) >= max_len:
        buffer.pop(0)
    buffer.append(predict)  
    smoother.smooth(buffer)
    smoothed_pred = smoother.smooth_data[0]
    return np.mean(smoothed_pred[-min(max_len, len(smoothed_pred)):])


def main():
    ser = initialize_uart()
    expansion_coefficient = 20
    temperature_upper_bound = 37
    valid_region_area_limit = 10
    data_shape = (24, 32)
    resize_dim = (640, 480)

    prepipeline = PrePipeline(expansion_coefficient, temperature_upper_bound, buffer_size=10, data_shape=data_shape)
    stage1procerss = TrackingDetectingMergeProcess(expansion_coefficient, valid_region_area_limit)
    roipooling = ROIPooling((200, 400), 100, 100)
    range_estimator = pickle.load(open('Models/hgbr_range2.sav', 'rb'))
    kalman_smoother = KalmanSmoother(component='level_trend', component_noise={'level': 0.0001, 'trend': 0.01})
    buffer_pred = {}

    print("Start!!")
    try:
        while True:
            try:
                data = ser.readline().strip()
            except:
                continue
            if data:
                try:
                    msg_str = data.decode('utf-8')
                except UnicodeDecodeError:
                    continue
                sensor_mat, sensor_at = preprocess_temperature_data(msg_str)
                if sensor_mat is None:
                    continue

                ira_img, _, ira_mat = prepipeline.Forward(np.flip(sensor_mat, 0), sensor_at)
                if not isinstance(ira_img, np.ndarray):
                    continue

                mask, _, filtered_mask_colored, _, _, _, valid_BBoxes, valid_timers = stage1procerss.Forward(ira_img)
                ira_colored = apply_color_map(SubpageInterpolating(np.flip(sensor_mat, 0)), expansion_coefficient, temperature_upper_bound, resize_dim)

                depth_map = np.zeros_like(filtered_mask_colored, dtype=float)
                for idx, (x, y, w, h) in enumerate(valid_BBoxes):
                    if not (100 < (x + w / 2) < 500):
                        continue

                    roi_t = ira_mat[y:y + h, x:x + w]
                    pooled_roi = roipooling.PoolingNumpy(roi_t)
                    input_data = np.concatenate([np.sort(pooled_roi.flatten())[::-1][:8], [x + w / 2, y + h / 2]])

                    predict_r = range_estimator.predict(input_data.reshape(1, -1))[0]
                    if idx in buffer_pred:
                        predict = smooth_predictions(buffer_pred[idx], kalman_smoother, predict_r)
                    else:
                        buffer_pred[idx] = [predict_r]
                        predict = predict_r

                    cv2.rectangle(ira_colored, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(ira_colored, f"{round(predict, 2)}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,  color=(0, 0, 255),  fontScale=1.2, thickness = 3)

                    center_pt = (int(y + h / 2), int(x + w / 2))
                    for m in RegionDivid(filtered_mask_colored):
                        if m[center_pt[0], center_pt[1]] > 0.1:
                            depth_map += m * predict
                            break

                depth_map = np.where(depth_map < 0.1, 4.5, depth_map)
                depth_colormap = cv2.applyColorMap(((depth_map / 4.5) * 255).astype(np.uint8), cv2.COLORMAP_JET)
                combined_image = np.hstack((ira_colored, depth_colormap))

                cv2.imshow('All', combined_image)
                if cv2.waitKey(1) & 0xFF in {27, 113}:  # Press 'Esc' or 'q' to quit
                    break

        ser.close()
        cv2.destroyAllWindows()
    except KeyboardInterrupt:
        ser.close()
        cv2.destroyAllWindows()
        print("Process interrupted by user")


if __name__ == "__main__":
    main()

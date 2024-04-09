# TADAR

This repository contains the code for the paper:

> TADAR: Thermal Array-based Detection and Ranging for Privacy-Preserving Human Sensing

![This is the caption\label{mylabel}](readme_figs/modality-compare3.png)
Modality Comparison. Thermal array sensors strike a balance between imaging resolution and privacy protection for ubiquitous human sensing. The IR image is
from the IR module of the Realsense D455 depth camera. The mmWave imaging is done by a radar with 20 × 20 antennas. The 3D point cloud is generated by TI IWR1843 radar.

![This is the caption\label{mylabel}](readme_figs/OverallDemoShown2.png)
We present TADAR, the first Thermal Array-based Detection and Ranging system that estimates the inherently missing range information for multi-user scenarios, extending thermal array outputs from 2D thermal pixels to 3D depths.

**The Demo Video is on https://www.youtube.com/watch?v=8l81C-WJqlE**

## Environment Setup
```
conda create -n tadar python=3.8.18
conda activate tadar
pip install -r requirements.txt
```

## Dataset

To obtain the dataset for this project, please access the provided [link](https://drive.google.com/drive/folders/1W6s4uIVd3ZRbgmoRzxf5zKAMinXRwjPW?usp=sharing) and save it in the root directory. The dataset consists of two files: Dataset.zip and Outputs.zip. Dataset.zip contains the original data gathered by the thermal array sensor, while Outputs.zip contains the results generated by DATAR.

## Running the Code

### Training the Thermal Ranging Model
```
python train.py
```
Note: The training process will generate the ranging model, which will be saved in the Models folder. This process may take some time. Alternatively, **you can use the pre-trained model provided in the Models folder**.

### Testing TADAR
```
python test.py
```
Note: The testing process will produce the detection and ranging results, which will be saved in the Outputs folder. This process may **take a significant amount of time**. We have provided the saved results in the link above, which **you can directly use to generate the main result figures mentioned in the paper**.

### Visualizing the Results
```
Run the visualization.ipynb
```
Note: The visualization.ipynb file will generate the main result figures mentioned in the paper.

## Citation

Coming soon.

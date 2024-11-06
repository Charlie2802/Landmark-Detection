# Facial Landmark Detection: 106-Point Regression Model

## Overview

This repository contains a regression-based model designed for **106 facial landmark detection**. The goal is to predict 106 (x, y) coordinates on human faces. Below is an example of the 106 landmarks used in our dataset.

<img src="https://gitlab.com/aaditya7822823/landmark-detection/-/raw/main/image.png" alt="Landmark Image" width="600"/>

## Directory Structure

### 1. TRAINING-DATA
The training data is split into multiple folders for various datasets and augmentations:

#### **Train_1**
- Combines **Chinese box images**, **office images**, and **pose images**.
- Dataset is split into:
  - **Train**: 18,000 images
  - **Test**: 2,000 images
  - **Validation**: 2,000 images

#### **Train_2_pdb**
- Contains an additional set of pose-detection images using a mean algorithm to calculate the difference from the mean pose.
- Augmentation techniques (random scaling, rotation) are applied to double the size of the dataset.
  - **Train**: 19,000 images
  - **Test**: 3,700 images
  - **Validation**: 3,700 images



<img src="https://gitlab.com/aaditya7822823/landmark-detection/-/raw/main/pdb_projection_histogram.png" alt="Pose Detection Histogram and Mean Pose" width="600"/>



#### **Train_3_pdb**
- Further augmentations (*3x) like random flipping, grayscale, etc., were applied.
  - **Train**: 22,000 images
  - **Test**: 2,200 images
  - **Validation**: 2,200 images

**Note:** Final augmentations are applied one-to-one to all images in the dataset.

#### Images
- This folder contains all the images used, combined using a `combined.csv` file.

### 2. TEST-VIDEO
- This folder contains testing face videos, as landmark detection is primarily used in real-time applications.
- Testing is conducted on **videos** to ensure the model's performance also to test the jitter of the model as welll how stable it is and on different poses.

### 3. TRAINED-MODELS
- This folder contains different trained models saved as **H5**,**PB**, **tfjs** and **tflite** files.
- **H5 vs PB**: H5 models are typically saved with the Keras API and are easier to manipulate, while PB files are TensorFlow frozen models optimized for inference.

### 4. Training-Files
#### **Model Architecture**
  This folder contains different models used, including:

  - **CNN6.py**: Basic model used in the Wing Loss paper. While it didn't perform well, it's often used as a baseline for comparison in landmark detection models (WFLW dataset).
    
  - **custom.py**: A depthwise convolutional model that performed efficiently on our dataset, but struggled with lip detection.

  - **eff_transformer.py**: TensorFlow implementation of the Efficient Transformer Block, providing lightweight but powerful transformer layers for feature extraction.

  - **hourglass.py**: Popular in recent papers, this model provides stacked hourglass networks for iterative refinement of landmarks. While it's theoretically strong for landmark detection, our results showed it underperformed.

  - **mobilenet_v3.py**: A basic MobileNet model, with the first 23 layers frozen. Decent results were achieved, but it was not as efficient as other models.

  - **squeezenet.py**: Known for having the lowest NME (as low as 0.04) and accuracy (up to 0.6), but the overall results were not competitive with other models like custom or Xception.

    Two versions of **SqueezeNet** were used:
      - SE_224
      - SE_512

#### **Pose_Detection_Bias**
This folder implements a pose detection algorithm, where we curated the dataset to improve the performance of **train_2** and **train_3**. The algorithm segregates images and augments them to perform better on pose images.

- **pdb.py**: Implements the algorithm to separate pose images from the main dataset.
- **augment_pdb.py**: Augments the pose images using scaling, rotation, and other techniques.

#### **Test**
- **test_img_aspect.py**: Used for testing images.
- **test_vid_aspect.py**: Used for testing videos.
  
The resizing function maintains the **aspect ratio** by adding borders to the images, which improves the performance compared to resizing without considering the aspect ratio.

#### **Train**
- **losses.py**: Contains various loss functions experimented with during the training.
  - **Wing Loss** performed better than traditional MAE.
  - We also implemented **weighted wing loss**, giving more weight to the lips area, which showed slight improvements.
  - **nme.py**: Script to calculate the Normalized Mean Error, an important evaluation metric for landmark detection models.

- **train_2.ipynb** and **train_3.ipynb**: 
  These Jupyter notebooks contain the training scripts for the models. Training is performed using TensorFlow with the following augmentations applied:
  
  - **Random Occlusion**
  - **Random Grayscale**
  - **Random Flipping**
  - **Random Rotation**
  - **Random Zoom**
  - **Random Crop**

  The training process also utilizes a learning rate reduction strategy with the following parameters:
  
  - **Initial Learning Rate**: Set to a specified value.
  - **Factor**: 0.5
  - **Patience**: 3 epochs
  - **Minimum Learning Rate**: ( 10^{-10}\)
  - **Verbose**: 1 (to display training logs)

  The loss function used is a combination of **Mean Absolute Error (MAE)** and **Wing Loss**, with different weights assigned to each component.

- **split.py**: to split the data into train, test and val 


### 6. TRAINED_MODELS
This folder contains all the trained models saved as:
  - **H5 files**
  - **PB files**
  - **TF Lite files**
  - **TFJS models**

Models follow a naming convention: `architecture_dataset_customloss_size`.

LIPS DIFF: *def*

# Model Comparison Table

| *Video*  | *Model Name*                                 | Input Size | Model Size | LIPS DIFF |
|---------|---------------------------------------------|------------|------------|-----------|
| Rushali | custom_2_v3_CelebA_WA_CA_v3_corrected_model | 512        | 6mb        | 4.06      |
| Rushali | custom_2_v3_CelebA_WA_CA_v3_corrected_model | 512        | 6mb        | 4.12      |
| Rushali | custom_2_v3_CelebA_WA_CA_v3_model           | 512        | 6mb        | 4.21      |
| Rushali | EffNetB4_v1_ADD                             | 300        | 200mb      | 4.89      |
| Rushali | EffNetB4_v1_ADD_OfficeClean                 | 300        | 200mb      | 4.61      |
| Rushali | EffNetB7_v1_ADD                             | 300        | 700mb      | 10.33     |
| Rushali | Xception_CelebA                             | 299        | 300mb      | 4.24      |
| Rushali | Xception_v3                                 | 299        | 250mb      | 2.96      |
| Rushali | Xception_v3_ADD                             | 299        | 250mb      | 3.07      |
| Rushali | Xception_v3_aug                             | 299        | 250mb      | 2.66      |
| Rushali | Xception_v3_model                           | 299        | 83mb       | 2.96      |
| Rushali | Xception_pdb3_weightedwind_299              | 299        | 84mb       | 2.96      |
| Rushali | custom_pdb3_data_weightedwing_2_224         | 224        | 7.5mb      | 7.21      |
| Rushali | custom_pdb3_data_weightedwing_3_224         | 224        | 7.5mb      | 8.46      |
| Rushali | custom_pdb3_data_weightedwing_4_224         | 224        | 7.5mb      | 9.53      |
| Rushali | custom_pdb4_data_weightedwing_2_224         | 225        | 7.5mb      | 11.02     |
| Rushali | custom_train1_data_weightedwing_2_224       | 226        | 7.5mb      | 8.05      |
| Rushali | TNN_112_v3_CelebA                           | 112        | 11mb       | 4.81      |
| Rushali | TNN_v3_112                                  | 112        | 11mb       | 6.99      |
| Rushali | TNN                                         | 256        | 5mb        | 8.3       |
| Rushali | TNN_256_CelebA_v                            | 256        | 5.3mb      | 5.3       |



### Future Work
- Improving pose detection and ensuring real-time performance on videos.
- Further tuning of loss functions to improve accuracy on specific regions like eyes and lips.

# Heartbeat Sounds Classification using Machine Learning
**Capstone Project**: Classifying heartbeat anomalies from recorded audio
*General Assembly Data Science Immersive DSI13
24 April 2020*


## Problem Statement

Stethoscope is good in detecting the first warning signs of heart disease. However, this is always done by frontline healthcare workers, mostly when we are sick or during our annual physical examination. Although digital stethoscopes is available but we still need healthcare professional to determine whether an individual is having hearbeat irregularities or abnormalities and need for further check-up. 
It is well known that early detection and intervention in heart disease (or any disease) will greatly imporves the lifespan and the effectiveness of treatment options. 

The **goal** of this project is to use machine learning methods to identify and **classify heartbeat sounds** from audio collected from stethoscopes into normal versus various non-normal heartbeat sounds.


## Executive Summary
This capstone project is a Multiclass Classification problem, where it came from Kaggle heartbeat sound challenge and the data source can be found in below link:
[kaggle dataset: Heartbeat Sounds](https://www.kaggle.com/kinguistics/heartbeat-sounds)

The scope of this project constraint to challenge 2, that is heartbeat classification. 

Classifier used in this project are 2D CNN (Convolutional Neural Networks) and XGBOOST (eXtreme Gradient Boosting). Two features extracted techniques are explore, MFCC (MEL-Frequency Cepstral Coefficients) and CWT (Continuous Wavelet Transform). 

The following sections are supported by the respective numbered of Jupyter Notebooks

1. [Exploratory Data Analysis (EDA)](#EDA)

2. [Data Preprocessing and Feature Extraction](#Data-Preprocessing-and-Feature-Extraction)

3. [Multiclass Classifier Models:](#Multiclass-Classifier-Models)
    - a. MFCC feature with CNN
    - b. CWT feature with CNN
    - c. CWT feature with XGBOOST
    
4. [Model Evaluation, Discussion, and Recommendations](#Model-Evaluation,-Discussion,-and-Recommendations)


## EDA

- Loading audio wav from two datasets (set_a & set_b) using `Librosa` (audio and music processing in Python)

|Dataset|Number of Categories|Categories|Sources|Recorded by|
|---|---|---|---|---|
|A|4|Normal, Murmur, Extra Heart Sound, Artifact|general public|iStethoscope Pro iPhone app|
|B|3|Normal, Murmur, Extrasystole|clinic trial in hospitals|digital stethoscope|

- audio files are in varying lengths, between 1 second and 30 seconds. 
- total of 5 categories and the categories distributions are as follows:

|Categories|%|
|---|---|
|artifact|6.8|
|extrahls|3.2|
|extrastole|7.9|
|murmur|22.1|
|normal|60.0


## Data Preprocessing and Feature Extraction
- First step is to denoise audio signal by using **Discrete Wavelet Transform** (DWT). Wavelet family used is '**Daubechies**’, subcategories of 'db6' and level 10.
- Next step is to perform **imbalance class** treatment on TRAIN data after **train/test split** in 2 steps: **Systhesizes new samples** and **Oversampling** minority class.
- Finally step is to extract the audio features to train the classification model. 
- First feature is **MFCC**, which is a popular techniques to extract feature from raw audio data. **20 cepstral coefficients** are extracted.
- The second feature is **CWT**, where it transform 1D signal into 2D time-scale representation of the signal in the form of a scaleogram. **Shannon wavelets** ('shan1.5-1.0') is chosen to perform the transformation.


## Multiclass Classifier Models

**Baseline Accuracy**
In EDA, it is observed that the dataset is with imbalance class, with `normal` heartbeat as the majority class of 60%. Thus, my naive baseline accuracy is **60%**. 

**Modelling Approach**
I explore 3 modelling approaches:
1. MFCC with CNN classifier
2. CWT with CNN classifier
3. CWT with XGBOOST classifier

**CNN**
The model consists of **3 x convolutional + pooling layer**. Max pooling is used and dropout is added in 2nd and 3rd convolutional layer. Next is **flatten** it and pass it to fully connected dense layer. Droupout, BatchNormalization, and Regularization is added to reduce overfitting. Output layer is wiht **5 neurons**. Activation for all layer is using '**relu**' (Rectified linear unit) except '**softmax**' for the output layer. 

**XBGOOST**
Prior to modelling the **CWT features**, **PCA** (Principal Components Analysis) is applied to extract the most important coefficients per period to feed into the XGBOOST Classifier. Ojective of **"multi:softmax"** is selected for multiclass classification.

**RandomizedGridSearchCV** is used to find the optimal hyperparameter to improve classificaitn efficiency of the models.


## Model Evaluation, Discussion, and Recommendations
**Metrics**
- Precision, f1, and accuracy scores. 

**Precision** provides the *positive predictive value, the proportion of samples that belong in category $a$ that are correctly placed in category $a$*. In the context of this project, among all predicted murmur heartbeat for example, how many did I predict correctly? High precision is with low FP (predict to be murmur but actually is not).

**f1** is **weighted average for precision and recall**. This score takes both FN and FP into account. It is more useful than accuracy in this project, particularly dataset is with uneven class distribution.

On top of that, Youden's Index, F-score of problematic heartbeats, and Discriminant Power of problematic heartbeats are used to evaluate the efficiency of the classifier model on the unlabel dataset, using the 'locked' excelsheet provided by kaggle.

### Results
Among the three approaches, **CWT + CNN** with imbalance class treatment perform the best. Though the accuracy is about **59%**, which at a glance, worst than the baseline accuracy. However, accuracy is not a good metric to acces efficiency of this imbalance multi-class classifier. The **CWT + CNN** achieves **80%** precision in detecting murmur, **73%** precision in detecting normal,and **20%** precision in detecting extra heart sound. It is also having more consistence results performance in classifying the unseen data (unlabel set_a and set_b). In addition, it perform better compared to a reference from a technical paper using the SAME datasets (unlabel set_a nad set_b).

However, the model's efficiency in classifying normal and various non-normal heartbeat sounds still need to further improve as the precision, f1 scores for some of the heatbeat category is still low. 

### Recommendation for the improvements
1. More data. The label datasets are only 585 and 60% of them are from normal heartbeat category. Thus, more data for both normal and non-normal heartbeat categories are needed to train the clasiffier model. Audio wav with at least 3 sec is desirable as some of the audio wav is as short as 0.7sec. 

2. Segmentation approach in preprocessing could be improved. This project use a segment of T=3sec for training and validation, where post-zero padding is apply on audio shorter than 3sec. Segmentation could be improve to first detect the position of heartbeat S1 & S2, and segment the audio into smaller chunck by locating the begining of the heartbeat and ensuring the segmented audio consist of at least x amount of heart cycles/ hearbeat pairs. 

3. CWT feature extraction optimization by identifying the least number of CWT coefficients that does not loose out the important details of the audio signal. This will reduce the computation rsources significantly in CNN.

4. Explore RNN (improve the model efficiency in classifying the heartbeat category) and XGBOOST (maintain the model efficiency but with much lesser computational time)


## Acknowledgements
- The Classifying Heart Sound Challenge from kaggle, which is sponsored by the PASCAL network of Excellence
- GA DSI13 instructor, tutors and my fellow classmates


## Reference
1. [Peter Bentley et al](http://www.peterjbentley.com/heartchallenge/)
2. [Guide using wavelet transformation in machine learning](http://ataspinar.com/2018/12/21/a-guide-for-using-the-wavelet-transform-in-machine-learning/)
3. [librosa.feature.mfcc](https://librosa.github.io/librosa/generated/librosa.feature.mfcc.html)
4. [mfcc](https://medium.com/prathena/the-dummys-guide-to-mfcc-aceab2450fd)
5. [Continuous Wavelet Transform (CWT)](https://pywavelets.readthedocs.io/en/latest/ref/cwt.html)
6. [A gentle introduction to wavelet for data analysis](https://www.kaggle.com/asauve/a-gentle-introduction-to-wavelet-for-data-analysis)
7. [For basic conversion between scales and time/frequency domain values by A. Sauve](https://github.com/alsauve/scaleogram/blob/master/doc/scale-to-frequency.ipynb)
8. [Beginner's Guide to Audio Data](https://www.kaggle.com/fizzbuzz/beginner-s-guide-to-audio-data)
9. [Sound Classification using Deep Learning](https://medium.com/@mikesmales/sound-classification-using-deep-learning-8bc2aa1990b7)
10. [Intro to Classification and Feature Selection with XGBoost](https://www.aitimejournal.com/@jonathan.hirko/intro-to-classification-and-feature-selection-with-xgboost)
11. [Multiple Time Series Classification by Using Continuous Wavelet Transform](https://towardsdatascience.com/multiple-time-series-classification-by-using-continuous-wavelet-transformation-d29df97c0442)
12. [A Robust Heart Sound Segmentation and Classification Algorithm using Wavelet Decomposition and Spectrogram](https://www.academia.edu/2787263/A_Robust_Heart_Sound_Segmentation_and_Classification_Algorithm_using_Wavelet_Decomposition_and_Spectrogram)
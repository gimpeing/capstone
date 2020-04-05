# GA DSI13 Capstone

## Overview

This is my captone project journey in GA DSI13

## Check point 1: Topic selections

Progress report: *16/3/2020* 

We need to propose **3 topics** for our capstone project and share it in the class (lightning talk) about:

1. The problem statement
2. Target audience
3. Success metric 
4. Data source 
5. Potential challenges or obstacles
6. Is this a reasonable project given the time constraints that you have?

My `proposed topics` are:
1. Heartbeat Sounds Classification (multiclass - classification)
2. Sentiment Analysis of Earphones Review (NLP and LDA topic analysis)
3. Prediction of Dengue Outbreak (Binary classification)

Details can be found in `*1_capstone_checkpoint1_topics.ipynb*`


## Check point 2: Confirm the topic for Capstone

Progress report: *26/3/2020*

Update on my chosen topic to work on.
Chosen topic: **`Heartbeat Sound Classification`**

Deliverables: 
- Firm up my problem statement, and the goal
- Outlines my proposed methods and models
- Defines the risks & assumptions of your data 
- Revises initial goals & success criteria, as needed
- Documents your data source
- Performs & summarizes preliminary EDA of your data

Details can be found in `*2_capstone_checkpoint2_eda.ipynb*`


### Obtacles:

*30/3/2020*

This is Kaggle project, which consist two challenges:

1. Identify the locations of heart sounds from the audio.
2. Classify the heart sounds into one of several categories (normal v. various non-normal heartbeat sounds -asthma, pneumonia and bronchiolitis).

I started to perform EDA for the first challenge (or task 1). I had issues in identifying what is the feature to extract in order to use machine learning to identify the locations of heartbeat sounds (i.e. S1 and S2) from normal heartbeat audio. 

I discussed my issues with instructor and decided to proceed to task 2 instead, which, the task is to classify the heartbeat sounds' category (normal and various non-normal heartbeat sounds). I could use RNN that is goingt o learn soon for this classification problem. With this suggestion, I proceed to concentrate on task 2. Previously, I was mis-interpreted that task 1 is the pre-requisite for task 2, which, actually, they can be a separate stand-alone topic.

Work done till this stage: `*3_capstone_s1s2detection_eda_preprocessing.ipynb*`

## Check point 3: Progress update for Capstone

Progress report: *6/4/2020*

At this stage, I should have data in hand and some models made.

I have completed the first modeling: Convolutional Neural Network (RNN). The accuracy score on test set is around 78%. I have yet to use the model to predict the unlabel dataset as there are still a lot of area to fine tune both the feature extraction and modeling hyperparameter tuning.
The model is not able to predict two minority classes, i.e. **extrahls** and **extrastole**, none of them predicted correctly.

Works to be carried out: 
1. revisit those audio wav that is wrongly classify and extra way to improve the audio wav pre-processing. 
2. explore the impact of wav duration use in feature extraction. The first model is with dropping audio wav <1sec, zero pad (post) on wav duration between 1-4sec. Post truncate wav >4sec.
3. explore any extra feature that can help to train the model to distinguish the minority class (extrahls, extrastole).

Details can be found in `*heartbeat_classification.ipynb*`

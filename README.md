# End-to-End Multi-class Dog Breed Classification

This project details the development of an end-to-end multi-class image classifier using **TensorFlow** and **TensorFlow Hub**.

\<a href="[https://colab.research.google.com/github/Olamaryse/Dog-vision/blob/main/dog\_vision.ipynb](https://colab.research.google.com/github/Olamaryse/Dog-vision/blob/main/dog_vision.ipynb)" target="\_parent"\>\<img src="[https://colab.research.google.com/assets/colab-badge.svg](https://colab.research.google.com/assets/colab-badge.svg)" alt="Open In Colab"/\>\</a\>

-----

## 1\. Problem

The primary objective of this project is to **identify the breed of a dog given an image** of the dog.

-----

## 2\. Data

The dataset utilized for this classification task is sourced from **Kaggle's dog breed identification competition**.

The dataset can be accessed at: [https://www.kaggle.com/competitions/dog-breed-identification/data](https://www.kaggle.com/competitions/dog-breed-identification/data)

-----

## 3\. Evaluation

The model's performance is evaluated based on a file containing **prediction probabilities for each dog breed for every test image**.

-----

## 4\. Features

Key characteristics of the data include:

  * The project involves **unstructured image data**, making deep learning and transfer learning suitable approaches.
  * There are **120 distinct dog breeds**, indicating a multi-class classification problem with 120 different classes.
  * The training set comprises **over 10,000 labeled images**.
  * The test set contains **over 10,000 unlabeled images**.

-----

## Environment Setup

The workspace was prepared by importing necessary libraries and verifying GPU availability.

  * **TensorFlow version:** 2.18.0
  * **TensorFlow Hub version:** 0.16.1
  * **GPU availability:** YES

-----

## Data Preparation

For machine learning models, data must be in a numeric format. The project involved loading and inspecting the `labels.csv` file, which contains the dog breed labels.

### Labels Data Description:

```
                                      id               breed
count                              10222               10222
unique                             10222                 120
top     fff43b07992508bc822f33d8ffd902ae  scottish_deerhound
freq                                   1                 126
```

### First 5 Rows of Labels Data:

```
                                 id             breed
0  000bec180eb18c7604dcecc8fe0dba07       boston_bull
1  001513dfcb2ffafc82cccf4d8bbaba97             dingo
2  001cdf01b096e06d78e9e5112d419397          pekinese
3  00214f311d5d2247d5dfe4fe24b2303d          bluetick
4  0021f9ceb3235effd7fcde7f7538ed62  golden_retriever
```

# Music Genre Recognition using Convolutional Neural Networks

## Overview
This project implements a Convolutional Neural Network (CNN) model for Music Genre Recognition using the GTZAN Dataset. The model is trained to classify audio tracks into 10 music genres. Additionally, the project provides a user-friendly visualization for learned filters in each CNN layer, which represents the network's current understanding of music genres.

## Network Architecture

<img width="705" alt="Screenshot 2024-05-13 at 19 13 46" src="https://github.com/timchan9742/music-genre-recognition/assets/167204379/d7e6e044-a226-44c8-8cd0-13db4768593d">

## Dataset
The GTZAN Dataset used in this project is available [here](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification). It consists of 1000 audio tracks, each 30 seconds long, evenly distributed across 10 music genres. 

## Results
The trained model achieved an accuracy of 78% on the GTZAN Dataset.
<br/><br/>
<img width="711" alt="Screenshot 2024-05-12 at 15 18 55" src="https://github.com/timchan9742/music-genre-recognition/assets/167204379/ddfcc4f8-d420-4f02-b737-adfeea9d3ca8">

## Directory Structure
- `/data`: This directory contains the preprocessed data in .pkl format.
- `/models`: This directory contains the trained CNN model.
- `/genres`: This directory contains the GTZAN Dataset.
- `/visualization`: This directory contains the filter visualizations from each CNN layer.

## Usage
1. Clone the repository.
2. Download the GTZAN Dataset and place it in the `/genres` directory.
3. Run feature_extraction.py for data preprocessing.
4. Run train.py to train the CNN model.
5. Test the model or visualize the learned filters using the provided scripts.

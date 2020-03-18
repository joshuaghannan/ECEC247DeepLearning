# ECE C247 Final Project Submission

In this project, we analyze the classification of EEG data from the BCI Competition using several Recurrent Neural Network (RNN) architectures. In particular, we study simple Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) structures, and investigate equipping these net- works with initial convolutional layers for increased accuracy. In addition to comparing network structures, we analyze several choices of data pre-processing, ranging from simple data windowing to wavelet transforms. We show that data windowing combined with an architecture composed of stacked CNN and RNN layers achieves 62% classification accuracy for Subject1 and 61% classification accuracy accross all subjects on the test data.

## Authors:

* Joshua Hannan
* Eden Haney
* Yiming Zhou
* Jonathan Bunton

## How to run the project:

This project was run using Google Colaboratory. Training, validation, and test data are placed in a Drive folder.
In order to run the project, perform the following steps:

1. Run pip install -r requirements.txt in your shell.
2. Open any of the Notebooks in `final_version/` in Google Colab (`main_pipeline.ipynb` is the main test pipeline)
3. Select the type of data aumentation and neural network to use
4. Run the project, giving google drive access when prompted

## Contents 

1. `final_version/main_pipeline.ipynb`: Main test pipeline
2. `final_version/waveform_tests_1.ipynb`: Waveform test examples 1
3. `final_version/waveform_tests_2.ipynb`: Waveform test examples 2
4. `final_version/waveform_tests_3.ipynb`: Waveform test examples 3
5. `final_version/models.py`: File that contains the models
6. `final_version/data_utils.py`: File that contains the data augmentation methods
7. `final_version/utils.py`: File that contains the training and testing functions
8. 'C247_Final_Report.pdf': Final project report
9. 'requirements.txt': Project package requirements


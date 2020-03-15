# EEG Classification with RNN
Rewrite a clean_pipeline.ipynb and import models, dataloaders and other utility functions from .py files

### models.py
Please define and add new models as a class here.

Now we have LSTMnet, GRUnet, CNNLSTMnet

### data_utils.py
The customized dataloader and data augmentation methods are defined here

### utils.py
The training validation and testing function are defined here

### indicating training on subject 1 only
If you want to train on subject 1 only:

Initializing dataset set sub_only = True and set sub_only = True in TestRNN()

Otherwise the model is trained on all the subjects.

If sub_only = False: return both acc (acc on all subjects) and sub_acc (acc on subject 1 only)

If sub_only = True: acc = sub_acc
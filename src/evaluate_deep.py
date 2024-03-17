import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
from keras.layers import Dense, Input, Flatten
from keras.models import Model
from glob import glob
import os
import argparse
from get_data_deep import get_data_deep
import matplotlib.pyplot as plt
from keras.applications.vgg19 import VGG19
import tensorflow
import seaborn as sns
from keras.models import load_model
import pandas as pd

def evaluate_model_deep(config_file):
    config = get_data_deep(config_file)
    batch = config['img_augment']['batch_size']
    class_mode = config['img_augment']['class_mode']
    te_set=config['model']['test_path']
    model = load_model('models/trained.h5')
    config=get_data_deep(config_file)

    test_gen = ImageDataGenerator(rescale=1./255)   # [0,1]
    test_set = test_gen.flow_from_directory(te_set,
                                                  target_size=(255,255),
                                                  batch_size=batch,
                                                  class_mode=class_mode)
    
    label_map = (test_set.class_indices)
    print(label_map)
    Y_pred = model.predict(test_set, len(test_set))
    y_pred = np.argmax(Y_pred, axis=1)
    print('Confusion MAtrix')
    sns.heatmap(confusion_matrix(test_set.classes, y_pred), annot=True)
    plt.xlabel('Actual Value, glioma: 0, meningioma: 1, notumor: 2, pituitary: 3')
    plt.ylabel('Predicted Value, glioma: 0, meningioma: 1, notumor: 2, pituitary: 3')
    plt.savefig('reports/Confusion Matrix')
    #plt.show()
    print('Classification Report')
    target_name=['glioma','meningioma','notumor','pituitary']
    df = pd.DataFrame(classification_report(test_set.classes, y_pred, target_names=target_name, output_dict=True)).T

    df['support'] = df.support.apply(int)

    df.style.background_gradient(cmap='viridis', subset=pd.IndexSlice['0':'9', 'f1-score'])

    df.to_csv('report/classification_report')
    print('Classification Report and Confusion Matrix Saved at Reports Directory')



if __name__ == '__main__':
    
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config',default='deep_params.yaml')
    passed_args = args_parser.parse_args()
    evaluate_model_deep(config_file=passed_args.config)
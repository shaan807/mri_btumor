import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Input, Flatten
from keras.models import Model
from glob import glob
import os
import argparse
from get_data_deep import get_data_deep
import matplotlib.pyplot as plt
from keras.applications.vgg19 import VGG19
import tensorflow
import mlflow
from urllib.parse import urlparse
from mlflow.keras import log_model

def train_model_deep_mlflow(config_file):
    config=get_data_deep(config_file)
    train = config['model']['trainable']
    if train == True:
        img_size = config['model']['image_size']
        trn_set = config['model']['train_path']
        te_set= config['model']['test_path']
        num_cls = config['load_data_deep']['num_classes']
        rescale = config['img_augment']['rescale']
        shear_range = config['img_augment']['shear_range']
        zoom_range= config['img_augment']['zoom_range']
        verticalf = config['img_augment']['vertical_flip']
        horizontalf = config['img_augment']['horizontal_flip']
        batch = config['img_augment']['batch_size']
        class_mode = config['img_augment']['class_mode']
        loss = config['model']['loss']
        optimizer = config['model']['optimizer']
        metrics = config['model']['metrics']
        epochs = config['model']['epochs']

        print(type(batch))

        resnet = VGG19(input_shape=img_size +[3], weights='imagenet', include_top=False)

        for p in resnet.layers:
            p.trainable = False # set T or F as per requirment

        op = Flatten()(resnet.output)
        prediction = Dense(num_cls, activation='softmax')(op)
        
        mod = Model(inputs = resnet.input, outputs=prediction)
        print(mod.summary())
        img_size = tuple(img_size)

        mod.compile(loss=loss, optimizer=optimizer, metrics=metrics)

        train_gen = ImageDataGenerator(rescale=1./255,
                                       shear_range=shear_range,
                                       zoom_range=zoom_range,
                                       horizontal_flip=horizontalf,
                                       vertical_flip=verticalf,
                                       rotation_range=90)
        test_gen = ImageDataGenerator(rescale=1./255)

        train_set = train_gen.flow_from_directory(trn_set,
                                                  target_size=(255,255),
                                                  batch_size=batch,
                                                  class_mode=class_mode)
        
        test_set = test_gen.flow_from_directory(te_set,
                                                  target_size=(255,255),
                                                  batch_size=batch,
                                                  class_mode=class_mode)

############################### MLFLOW STARTS FROM HERE #######################################
        mlflow_config_deep = config["mlflow_config_deep"]
        remote_server_uri_deep = mlflow_config_deep["remote_server_uri_deep"]
        mlflow.set_tracking_uri(remote_server_uri_deep)
        mlflow.set_experiment(mlflow_config_deep["experiment_name_deep"])

        with mlflow.start_run(run_name=mlflow_config_deep["run_name_deep"]) as mlops_run_deep:
            history = mod.fit(train_set,
                          epochs=epochs,
                          validation_data=test_set,
                          steps_per_epoch=len(train_set),
                          validation_steps=len(test_set)
            )

            train_loss = history.history['loss'][-1]
            train_acc = history.history['accuracy'][-1]
            val_loss = history.history['val_loss'][-1]
            print("train_loss:", train_loss)
            print("train_acc:", train_acc)
            print("val_loss:", val_loss)

            mlflow.log_param("epochs", epochs)
            mlflow.log_param("val_loss", val_loss)
            mlflow.log_param("train_loss", train_loss)
            mlflow.log_param("train_accuracy", train_acc)

            tracking_url_type_store_deep = urlparse(mlflow.get_artifact_uri()).scheme

            if tracking_url_type_store_deep!="file":
                mlflow.keras.log_model(mod, "model", registered_model_name=mlflow_config_deep["registered_deep_model_name"])
            else:
                mlflow.keras.log_model(mod, "model")
    else:
        print("Model is not trained by Xerxexz Solutions")


if __name__ == '__main__':
    
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config',default='deep_params.yaml')
    passed_args = args_parser.parse_args()
    train_model_deep_mlflow(config_file=passed_args.config)
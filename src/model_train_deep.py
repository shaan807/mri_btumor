# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import yaml

# # Load parameters from params.yaml file
# with open('deep_params.yaml') as f:
#     params = yaml.safe_load(f)

# # Extract parameters
# train_path = params['model']['train_path']
# test_path = params['model']['test_path']
# image_size = params['model']['image_size']
# epochs = params['model']['epochs']
# batch_size = params['img_augment']['batch_size']

# # Define image dimensions and other parameters
# img_width, img_height = image_size[0], image_size[1]
# input_shape = (img_width, img_height, 3)

# # Data Augmentation and Preprocessing
# train_datagen = ImageDataGenerator(
#     rescale=params['img_augment']['rescale'],
#     shear_range=params['img_augment']['shear_range'],
#     zoom_range=params['img_augment']['zoom_range'],
#     horizontal_flip=params['img_augment']['horizontal_flip'],
#     vertical_flip=params['img_augment']['vertical_flip']
# )
# test_datagen = ImageDataGenerator(rescale=params['img_augment']['rescale'])

# train_generator = train_datagen.flow_from_directory(
#     train_path,
#     target_size=(img_width, img_height),
#     batch_size=batch_size,
#     class_mode=params['img_augment']['class_mode']
# )

# test_generator = test_datagen.flow_from_directory(
#     test_path,
#     target_size=(img_width, img_height),
#     batch_size=batch_size,
#     class_mode=params['img_augment']['class_mode']
# )

# # Building the CNN model
# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
#     MaxPooling2D((2, 2)),
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),
#     Conv2D(128, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),
#     Flatten(),
#     Dense(512, activation='relu'),
#     Dropout(0.5),
#     Dense(params['load_data_deep']['num_classes'], activation='softmax')
# ])

# # Compile the model
# model.compile(optimizer=params['model']['optimizer'],
#               loss=params['model']['loss'],
#               metrics=params['model']['metrics'])

# # Train the model
# model.fit(train_generator, epochs=epochs, validation_data=test_generator)

# # Save the model
# model.save(params['model']['save_dir'])














# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# # Define image dimensions and other parameters
# img_width, img_height = 150, 150
# input_shape = (img_width, img_height, 3)
# epochs = 1
# batch_size = 32

# # Data Augmentation and Preprocessing
# train_datagen = ImageDataGenerator(rescale=1./255)
# test_datagen = ImageDataGenerator(rescale=1./255)

# train_generator = train_datagen.flow_from_directory(
#     'C:\\Users\\91890\\Desktop\\training\\data\\processed\\Training',
#     target_size=(img_width, img_height),
#     batch_size=batch_size,
#     class_mode='categorical')

# test_generator = test_datagen.flow_from_directory(
#     'C:\\Users\\91890\\Desktop\\training\\data\\processed\\Testing',
#     target_size=(img_width, img_height),
#     batch_size=batch_size,
#     class_mode='categorical')

# # Building the CNN model
# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
#     MaxPooling2D((2, 2)),
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),
#     Conv2D(128, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),
#     Flatten(),
#     Dense(512, activation='relu'),
#     Dropout(0.5),
#     Dense(4, activation='softmax')
# ])

# # Compile the model
# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])

# # Train the model
# model.fit(train_generator, epochs=epochs, validation_data=test_generator)

# # Save the model
# model.save('brain_tumor_classifier.h5')

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load parameters from param.yaml
import yaml

with open("deep_params.yaml", "r") as f:
    params = yaml.safe_load(f)

img_width, img_height = params["model"]["image_size"]
epochs = params["model"]["epochs"]
batch_size = params["img_augment"]["batch_size"]

# Data Augmentation and Preprocessing
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    params["model"]["train_path"],
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    params["model"]["test_path"],
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

# Building the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')
])

# Compile the model
model.compile(optimizer=params["model"]["optimizer"],
              loss=params["model"]["loss"],
              metrics=params["model"]["metrics"])

# Train the model
model.fit(train_generator, epochs=epochs, validation_data=test_generator)

# Save the model
model.save('trained1.h5')

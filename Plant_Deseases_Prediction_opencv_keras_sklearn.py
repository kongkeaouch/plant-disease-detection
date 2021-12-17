import os
import random
import matplotlib.pyplot as plt
import pandas as pd
from keras.layers import Activation, Flatten, Dropout, Dense
import cv2
import numpy as np
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import img_to_array, array_to_img
from sklearn.preprocessing import label_binarize, LabelBinarizer
from keras.utils import to_categorical
from keras.models import model_from_json
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing import image
from PIL import Image
from os import listdir
from matplotlib.image import imread
from google.colab import drive

drive.mount('/content/drive')
plt.figure(figsize=(12, 12))
path = '/content/drive/kongkea/plant/potato_early_blight'
for i in range(1, 17):
    plt.subplot(4, 4, i)
    plt.tight_layout()
    rand_img = imread(path+'/'+random.choice(sorted(os.listdir(path))))
    plt.imshow(rand_img)
    plt.xlabel(rand_img.shape[1], fontsize=10)
    plt.ylabel(rand_img.shape[0], fontsize=10)


def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None:
            image = cv2.resize(image, (256, 256))
            return img_to_array(image)
        else:
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None


dir = '/content/drive/kongkea/plant'
root_dir = listdir(dir)
image_list, label_list = [], []
labels = ['corn_rust',
          'potato_blight', 'tomato_bacteria']
binary_labels = [0, 1, 2]
temp = -1
for directory in root_dir:
    plant_image_list = listdir(f"{dir}/{directory}")
    temp += 1
    for files in plant_image_list:
        image_path = f"{dir}/{directory}/{files}"
        image_list.append(convert_image_to_array(image_path))
        label_list.append(binary_labels[temp])
label_counts = pd.DataFrame(label_list).value_counts()
label_counts.head()
image_list[0].shape
label_list = np.array(label_list)
label_list.shape
x_train, x_test, y_train, y_test = train_test_split(
    image_list, label_list, test_size=0.2, random_state=10)
x_train = np.array(x_train, dtype=np.float16)/225.0
x_test = np.array(x_test, dtype=np.float16)/225.0
x_train = x_train.reshape(-1, 256, 256, 3)
x_test = x_test.reshape(-1, 256, 256, 3)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(256, 256, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(0.0001), metrics=['accuracy'])
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.2)
epochs = 50
batch_size = 128
history = model.fit(x_train, y_train, batch_size=batch_size,
                    epochs=epochs, validation_data=(x_val, y_val))
model.save('/content/drive/kongkea/plant_disease.h5')
json_model = model.to_json()
with open('/content/drive/kongkea/plant_model.json', 'w')as json_file:
    json_file.write(json_model)
model.save_weights('/content/drive/kongkea/plant_model_weights.h5')
plt.figure(figsize=(12, 5))
plt.plot(history.history['accuracy'], color='r')
plt.plot(history.history['val_accuracy'], color='b')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'val'])
plt.show()
print('Computing model accuracy')
scores = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {scores[1]*100}")
y_pred = model.predict(x_test)
img = array_to_img(x_test[10])
img
print('Original: ', labels[np.argmax(y_test[10])])
print('Prediction : ', labels[np.argmax(y_pred[10])])

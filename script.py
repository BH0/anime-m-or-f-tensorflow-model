import numpy as np
import matplotlib.pyplot as plt
import os
import cv2 

DATADIR = "anime-genders"
CATEGORIES = ['Male', 'Female'] 

for category in CATEGORIES:
  path = os.path.join(DATADIR, category)
  for img in os.listdir(path):
    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
    break
  break

IMG_SIZE = 50
new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

training_data = []

def create_training_data():
  for category in CATEGORIES: 
    path = os.path.join(DATADIR, category)
    class_num = CATEGORIES.index(category)
    for img in os.listdir(path):
      try: 
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        training_data.append([new_array, class_num])
      except Exception as e: 
        # print(e)
        pass
    

create_training_data()

X = []
y = []

for features, label in training_data:
  X.append(features)
  y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

import pickle 
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)


from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten 
from tensorflow.keras.layers import Conv2D, MaxPooling2D
#from tensforflow.keras.callbacks import TensorBoard 
import pickle 
import time 

pickle_in = open("X.pickle", "rb")
pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)
y = np.array(y)
X = X/255.0
dense_layers = [0]
layer_sizes = [64] 
conv_layers = [3] 

model = Sequential()
model.add(Conv2D(26, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2))) 
model.add(Conv2D(26, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2))) 
model.add(Flatten())
model.add(Dense(52))
model.add(Activation('relu'))
model.add(Dense(len(CATEGORIES)))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, batch_size=32, epochs=30, validation_split=0.3)

model.save('anime-male-or-female.model')

def prepare(filepath):
  IMG_SIZE = 50
  img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE) 
  new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
  return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# predictions = model.predict([prepare('aguy.jpg')])
# predictions = model.predict([prepare('afemale.PNG')])
# predictions = model.predict([prepare('afemale.jpg')])
predictions = model.predict([prepare('afemale2.jpg')])
import tensorflow as tf
score = tf.nn.softmax(predictions[0])

print("The gender is likely {} with a {:.2f} percent confidence.".format(CATEGORIES[np.argmax(score)], 100 * np.max(score)))

import numpy as np
import tensorflow as tf
import keras
from keras.models import load_model



import matplotlib.pyplot as plt

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

train_images = train_images.reshape(60000, 28, 28, 1)
test_images = test_images.reshape(10000, 28, 28, 1)

try:
    loaded_model = load_model("/home/pi/model.h5")
    print("Loaded model from disk")
except:
    loaded_model = load_model("model.h5")
    print("Loaded model from disk")

loaded_model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

index = 0

X = test_images
Y = test_labels

scores = loaded_model.evaluate(X, Y)
print(scores)
print(loaded_model.metrics_names)

tmp = loaded_model.predict(X)

print('label:', train_labels[index], ':', class_names[train_labels[index]])
print('predict:', np.argmax(tmp[index]),  ':', class_names[np.argmax(tmp[0])])

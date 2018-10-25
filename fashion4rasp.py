import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import model_from_json

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

train_images = train_images.reshape(60000, 28, 28, 1)
test_images = test_images.reshape(10000, 28, 28, 1)

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

loaded_model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

X = test_images[:1,:,:,:]
Y = test_labels[:1]

# print(X,Y)
# scores = model.evaluate(X, Y)
tmp = loaded_model.predict(X)
scores = np.argmax(tmp)
print(tmp.shape, scores)
print(np.argmax(tmp[0]))
# print("%.2f%%" % (scores[1]*100))
import tensorflow as tf
import keras
from keras.callbacks import ModelCheckpoint, TensorBoard


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

train_images = train_images.reshape(60000, 28, 28, 1)
test_images = test_images.reshape(10000, 28, 28, 1)

model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation=tf.nn.relu, input_shape=(28, 28, 1)),
    keras.layers.Flatten(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])

model.summary()

pa = 'val_loss'
pa = 'val_acc'
ck_dir = "checkpoints/epoch_{epoch:02d}-"+pa+"_{"+pa+":.4f}.hdf5"
checkpoint = ModelCheckpoint(ck_dir, monitor=pa, verbose=1, save_best_only=True, save_weights_only=False, mode='max' if pa == 'val_acc' else 'min' if pa == 'val_loss' else 'auto', period=10)

tensorboard = TensorBoard(log_dir='./Graph', histogram_freq=20, write_graph=False, write_images=True)

history = model.fit(train_images, train_labels, batch_size=512, validation_data=(test_images, test_labels), epochs=100, callbacks=[checkpoint, tensorboard], verbose=0)

scores = model.evaluate(test_images, test_labels, verbose=0)
print(model.metrics_names)
print(scores)

# model.save("model.h5")
# print("Saved model to disk")
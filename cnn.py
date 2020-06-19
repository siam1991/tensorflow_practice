import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

(train_img, train_label), (test_img, test_label) = tf.keras.datasets.fashion_mnist.load_data()

train_img = train_img/255
test_img = test_img/255

train_img = np.expand_dims(train_img, -1)
test_img = np.expand_dims(test_img, -1)
print(train_img.shape)

# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'))
# print(model.output)
# model.add(tf.keras.layers.MaxPool2D())
# print(model.output)
# model.add(tf.keras.layers.Conv2D(64, (3, 3,), activation='relu'))
# print(model.output)
# model.add(tf.keras.layers.GlobalAveragePooling2D())
# print(model.output)
# model.add(tf.keras.layers.Dense(10, activation='softmax'))
# print(model.output)

"""训练模型"""
"""
model = tf.keras.Sequential([tf.keras.layers.Conv2D(64, (3, 3), input_shape=(28, 28, 1), activation='relu'),
                             tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                             tf.keras.layers.MaxPool2D(),
                             tf.keras.layers.Dropout(0.5),
                             tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
                             tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
                             tf.keras.layers.MaxPool2D(),
                             tf.keras.layers.Dropout(0.5),
                             tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
                             tf.keras.layers.GlobalAveragePooling2D(),
                             tf.keras.layers.Dense(64, activation='relu'),
                             tf.keras.layers.Dense(10, activation='softmax')])
print(model.summary())
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_img, train_label, epochs=20, validation_data=(test_img, test_label))
model.save('./model/fashion_mnist_cnn.h5')
model.save('./weights/cnn/fashion_mnist')
print(history.history.keys())

plt.plot(history.epoch, history.history['accuracy'])
plt.plot(history.epoch, history.history['val_accuracy'])
plt.show()

plt.plot(history.epoch, history.history['loss'])
plt.plot(history.epoch, history.history['val_loss'])
plt.show()
"""

model = tf.keras.models.load_model('./model/fashion_mnist_cnn.h5')
history = model.fit(train_img, train_label, epochs=10, validation_data=(test_img, test_label))
model.save('./model/fashion_mnist_cnn.h5')
model.save('./weights/cnn/fashion_mnist')
print(history.history.keys())
plt.plot(history.epoch, history.history['accuracy'])
plt.plot(history.epoch, history.history['val_accuracy'])
plt.show()

plt.plot(history.epoch, history.history['loss'])
plt.plot(history.epoch, history.history['val_loss'])
plt.show()



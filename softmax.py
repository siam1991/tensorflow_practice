import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

(train_img, train_label), (test_img, test_label) = tf.keras.datasets.fashion_mnist.load_data()
print(train_img.shape)
print(train_label.shape)
print(test_img.shape)
print(test_label.shape)

# plt.imshow(train_img[0])
# plt.show()
# print(train_label[0])
"""数据归一化"""
train_img = train_img/255
test_img = test_img/255
"""label 为顺序数字"""
model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)),
                             tf.keras.layers.Dense(128, activation='relu'),
                             tf.keras.layers.Dense(10, activation='softmax')])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_img, train_label, epochs=20)
model.save('./model/fashion_mnist.h5')
model.save_weights('./weights/fashion_mnist/fashion_mnist')
evaluate = model.evaluate(test_img, test_label)
print(evaluate)

"""label 转换为onehot形式"""
train_label_onehot = tf.keras.utils.to_categorical(train_label)
print(train_label_onehot[0])

test_label_onehot = tf.keras.utils.to_categorical(test_label)
print(test_label_onehot[0])

model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)),
                             tf.keras.layers.Dense(128, activation='relu'),
                             tf.keras.layers.Dense(10, activation='softmax')])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_img, train_label_onehot, epochs=20)
model.save('./model/fashion_mnist_onehot.h5')
model.save_weights('./weights/fashion_mnist_onehot/fashion_mnist_onehot')
evaluate = model.evaluate(test_img, test_label_onehot)
print(evaluate)




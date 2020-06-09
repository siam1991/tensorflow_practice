import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

mnist = np.load('./data/mnist.npz')
print(mnist.files)
x_train = mnist['x_train']
y_train = mnist['y_train']
x_test = mnist['x_test']
y_test = mnist['y_test']
print(x_train.shape)
print(y_train.shape)
# print(x_train[0])
# print(y_train[0])
# plt.imshow(x_train[0])
# plt.show()
x_train = x_train/255
x_test = x_test/255


"""label 为数字, loss使用sparse_catgorical_crossentropy"""
model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)),
                             tf.keras.layers.Dense(128, activation='relu'),
                             tf.keras.layers.Dense(10, activation='softmax')])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train, y_train, epochs=20)
model.save('./model/mnist.h5')
model.save_weights('./weights/mnist/mnist')
evaluate = model.evaluate(x_test, y_test)
print(evaluate)

"""label 为onehot，loss使用categorical_crossentropy"""
y_train_onehot = tf.keras.utils.to_categorical(y_train)
y_test_onehot = tf.keras.utils.to_categorical(y_test)
model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(28,28)),
                             tf.keras.layers.Dense(128, activation='relu'),
                             tf.keras.layers.Dense(10, activation='softmax')])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train_onehot, epochs=20)
model.save_weights('./weights/mnist_onehot/mnist_onehot')
model.save('./model/mnist_onehot.h5')
evalute = model.evaluate(x_test, y_test_onehot)

"""函数式api"""
input = tf.keras.Input(shape=(28,28))
x0 = tf.keras.layers.Flatten()(input)
x1 = tf.keras.layers.Dense(128, activation='relu')(x0)
output = tf.keras.layers.Dense(10, activation='softmax')(x1)
model2 = tf.keras.Model(inputs=input, outpus=output)
print(model2.summary())
model2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model2.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test))
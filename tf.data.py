import tensorflow as tf
import numpy as np
print(tf.__version__)

"""fashion mnist"""
(train_img, train_label), (test_img, test_label) = tf.keras.datasets.fashion_mnist.load_data()
train_img = train_img/255
test_img = test_img/255

# ds_train_img = tf.data.Dataset.from_tensor_slices(train_img)
# print(ds_train_img)
# ds_train_label = tf.data.Dataset.from_tensor_slices(train_label)
# print(ds_train_label)
# dataset = tf.data.Dataset.zip((ds_train_img, ds_train_label))
# print(dataset)
# ds_test_img = tf.data.Dataset.from_tensor_slices(test_img)
# print(ds_test_img)
# ds_test_label = tf.data.Dataset.from_tensor_slices(test_label)
# print(ds_test_label)
# dataset_test = tf.data.Dataset.zip((ds_test_img, ds_test_label))
# print(dataset_test)

dataset = tf.data.Dataset.from_tensor_slices((train_img, train_label))
print(dataset)
dataset_test = tf.data.Dataset.from_tensor_slices((test_img, test_label))
print(dataset_test)
batch_size = 64
dataset = dataset.shuffle(train_img.shape[0]).repeat().batch(batch_size)
dataset_test = dataset_test.batch(batch_size)
model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)),
                             tf.keras.layers.Dense(128, activation='relu'),
                             tf.keras.layers.Dense(10, activation='softmax')])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
steps_per_epoch = train_img.shape[0]//batch_size
validation_steps = test_img.shape[0]//batch_size
model.fit(dataset, epochs=20, steps_per_epoch=steps_per_epoch,
          validation_data=dataset_test,
          validation_steps=validation_steps)

"""mnist dataset"""
mnist = np.load('./data/mnist.npz')
x_train = mnist['x_train']
y_train = mnist['y_train']
x_test = mnist['x_test']
y_test = mnist['y_test']
print(x_train.shape, y_train.shape)
x_train = x_train/255
x_test = x_test/255

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
print(dataset)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
print(test_dataset)
batch_size = 64
dataset = dataset.shuffle(x_train.shape[0]).repeat().batch(batch_size)
test_dataset = test_dataset.batch(batch_size)
steps_per_epoch = x_train.shape[0]//batch_size
validation_steps = x_test.shape[0]//batch_size
model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)),
                             tf.keras.layers.Dense(128, activation='relu'),
                             tf.keras.layers.Dense(10, activation='softmax')])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(dataset, epochs=20, steps_per_epoch=steps_per_epoch, validation_data=test_dataset,
          validation_steps=validation_steps)
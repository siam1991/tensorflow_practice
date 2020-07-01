import tensorflow as tf
from datetime import datetime
import os

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_images = tf.expand_dims(train_images, -1)
test_images = tf.expand_dims(test_images, -1)

train_images = tf.cast(train_images/255, tf.float32)
test_images = tf.cast(test_images/255, tf.float32)

train_labels = tf.cast(train_labels, tf.int64)
test_labels = tf.cast(test_labels, tf.int64)

dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
dataset = dataset.repeat().shuffle(60000).batch(128)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
test_dataset = test_dataset.repeat().batch(128)

model = tf.keras.Sequential([tf.keras.layers.Conv2D(64, (3, 3), input_shape=(28, 28, 1), activation='relu'),
                             tf.keras.layers.Conv2D(64, (3, 3), activation= 'relu'),
                             tf.keras.layers.GlobalMaxPooling2D(),
                             tf.keras.layers.Dense(10, activation='softmax')])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
steps_per_epoch = 60000//128
validation_steps = 60000//128

log_dir = os.path.join('log', datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
tensor_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
file_writer = tf.summary.create_file_writer(log_dir+'/metrics')
file_writer.set_as_default()


def learning_rate_scheduler(epoch):
    learning_rate = 0.2
    if epoch > 5:
        learning_rate = 0.1
    if epoch > 10:
        learning_rate = 0.01
    if epoch > 20:
        learning_rate = 0.005
    tf.summary.scalar("learning rate", data=learning_rate, step=epoch)
    return learning_rate


learning_rate_callback = tf.keras.callbacks.LearningRateScheduler(learning_rate_scheduler)

model.fit(dataset, epochs=30, validation_data=test_dataset, validation_steps=validation_steps,
          steps_per_epoch=steps_per_epoch, callbacks=[tensor_callback, learning_rate_callback])



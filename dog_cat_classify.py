import tensorflow as tf
import pathlib
import random
import matplotlib.pyplot as plt

dc_dir = pathlib.Path('./data/dc_2000')

train_images_path = [str(path) for path in dc_dir.glob('train/*/*')]
print(train_images_path[:5])
random.shuffle(train_images_path)
train_labels_name = [path.split('\\')[-1].split('.')[0] for path in train_images_path]
print(train_labels_name[:5])

test_images_path = [str(path) for path in dc_dir.glob('test/*/*')]
test_labels_name = [path.split('\\')[-1].split('.')[0] for path in test_images_path]
train_size = len(train_labels_name)
print(train_size)
test_size = len(test_labels_name)
label_names = sorted(set(train_labels_name))
label_to_index = dict((label, index) for index, label in enumerate(label_names))
index_to_label = dict((v, k) for k, v in label_to_index.items())
print(label_to_index)
print(index_to_label)
train_labels = [label_to_index[label] for label in train_labels_name]
test_labels = [label_to_index[label] for label in test_labels_name]


def load_image(image_path):
    image_tensor = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image_tensor, channels=3)
    image = tf.image.resize(image, (200, 200))
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    return image


# 图片大小不一
# for i in range(5):
#     image = load_image(train_images_path[i])
#     plt.imshow(image)
#     plt.show()
#     print(image.shape)
#     image_tensor = tf.io.read_file(train_images_path[i])
#     image = tf.image.decode_jpeg(image_tensor, channels=3)
#     print(image.shape)


image_paths_dataset = tf.data.Dataset.from_tensor_slices(train_images_path)
image_dataset = image_paths_dataset.map(load_image)
train_labels_dataset = tf.data.Dataset.from_tensor_slices(tf.cast(train_labels, tf.int64))
train_dataset = tf.data.Dataset.zip((image_dataset, train_labels_dataset))

test_imgae_paths_dataset = tf.data.Dataset.from_tensor_slices(test_images_path)
test_image_dataset = test_imgae_paths_dataset.map(load_image)
test_labels_dataset = tf.data.Dataset.from_tensor_slices(test_labels)
test_dataset = tf.data.Dataset.zip((test_imgae_paths_dataset, test_labels_dataset))

batch_size = 128
train_dataset = train_dataset.repeat().shuffle(train_size).batch(batch_size)
test_dataset = test_dataset.repeat().batch(batch_size)
model = tf.keras.Sequential([tf.keras.layers.Conv2D(64, (3, 3), input_shape=(200, 200, 3), activation='relu'),
                             tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                             tf.keras.layers.MaxPool2D(),
                             # tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
                             # tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
                             # tf.keras.layers.MaxPool2D(),
                             # tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
                             # tf.keras.layers.Conv2D(236, (3, 3), activation='relu'),
                             # tf.keras.layers.MaxPool2D(),
                             tf.keras.layers.Flatten(),
                             tf.keras.layers.Dense(256, activation='relu'),
                             tf.keras.layers.Dense(1, activation='sigmoid')])

print(model.summary())
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
steps_per_epoch = train_size//batch_size
validation_steps = test_size//batch_size
history = model.fit(train_dataset, epochs=20, steps_per_epoch=steps_per_epoch,
                    validation_data= test_dataset, validation_steps=validation_steps)
model.save('./model/dc/dc.h5')
model.save('./weights/dc/dc')

plt.plot(history.epoch, history.history['accuracy'])
plt.plot(history.epoch, history.history['val_accuracy'], color='red')
plt.show()


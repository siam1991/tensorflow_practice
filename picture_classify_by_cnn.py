import tensorflow as tf
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import random

data_root = pathlib.Path('./data/2_class')
print(data_root)
all_image_paths = [str(path) for path in data_root.glob('*/*')]
random.shuffle(all_image_paths)
print(all_image_paths[:5])
label_names = sorted([item.name for item in data_root.glob('*/') if item.is_dir()])
print(label_names)
label_to_index = dict((label, i) for i, label in enumerate(label_names))
print(label_to_index)
index_to_label = dict((v, k) for k, v in label_to_index.items())
print(index_to_label)
all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]
print(all_image_labels[:5])


def load_image(image_path):
    image_tensor = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image_tensor, channels=3)
    image = tf.image.resize(image, [256, 256])
    image = tf.cast(image, tf.float32)
    image = image/255.0
    return image


for i in range(3):
    img_path = random.choice(all_image_paths)
    image = load_image(img_path)
    print(image.shape)
    print(pathlib.Path(img_path).parent.name)
    # plt.imshow(image)
    # plt.show()

image_path_dataset = tf.data.Dataset.from_tensor_slices(all_image_paths)
print(image_path_dataset)
image_dataset = image_path_dataset.map(load_image)
print(image_dataset)
label_dataset = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
dataset = tf.data.Dataset.zip((image_dataset, label_dataset))
print(dataset)
image_count = len(all_image_paths)
test_size = int(0.2*image_count)
train_size = image_count-test_size
batch_size = 32
train_dataset = dataset.skip(test_size)
test_dataset = dataset.take(test_size)

train_dataset = train_dataset.repeat().shuffle(train_size).batch(batch_size)
test_dataset = test_dataset.repeat().batch(batch_size)

# 构建模型
# cnn论文中建议batchNormalize 在卷积与激活之间，但实际使用情况，batchNormalize在激活之后比较好
# 若按照论文写法，则应该代码如下：
"""
tf.keras.layers.Conv2D(64, (3, 3)),
tf.keras.layers.BatchNormalization(),
tf.keras.layers.Activation('relu'),
"""
model = tf.keras.Sequential([tf.keras.layers.Conv2D(64, (3, 3), input_shape=(256, 256, 3), activation='relu'),
                             tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                             tf.keras.layers.BatchNormalization(),
                             tf.keras.layers.MaxPool2D(),
                             tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
                             tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
                             tf.keras.layers.BatchNormalization(),
                             tf.keras.layers.MaxPool2D(),
                             tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
                             tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
                             tf.keras.layers.BatchNormalization(),
                             tf.keras.layers.MaxPool2D(),
                             tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
                             tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
                             tf.keras.layers.BatchNormalization(),
                             tf.keras.layers.MaxPool2D(),
                             tf.keras.layers.Conv2D(1024, (3, 3), activation='relu'),
                             tf.keras.layers.BatchNormalization(),
                             tf.keras.layers.GlobalAveragePooling2D(),
                             tf.keras.layers.Dense(1024, activation='relu'),
                             tf.keras.layers.Dense(512, activation='relu'),
                             tf.keras.layers.Dense(1, activation='sigmoid')])
print(model.summary())
steps_per_epoch = train_size//batch_size
validation_steps = test_size//batch_size
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(train_dataset, epochs=20, validation_data=test_dataset,
                    steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)
model.save('./model/2_class_cnn.h5')
model.save('./weights/2_class/2_class_cnn')
print(history.history.keys())
plt.plot(history.epoch, history.history['accuracy'])
plt.plot(history.epoch, history.history['val_history'])
plt.show()

plt.plot(history.epoch, history.history['loss'])
plt.plot(history.epoch, history.history['val_loss'])
plt.show()
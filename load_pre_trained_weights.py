import tensorflow as tf
import glob

train_image_paths = glob.glob('./data/dc_2000/train/*/*')
print(train_image_paths[:5])
train_labels = [int(path.split('\\')[1] == 'cat') for path in train_image_paths]
print(train_labels[:5])

test_image_paths = glob.glob('./data/dc_2000/test/*/*')
print(test_image_paths[:5])
test_labels = [int(path.split('\\')[1] == 'cat') for path in test_image_paths]
print(test_labels[:5])


def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image)
    image = tf.image.resize(image, (256, 256))
    image = tf.cast(image, tf.float32)
    image = image/255
    return image


train_image_path_dataset = tf.data.Dataset.from_tensor_slices(train_image_paths)
train_image_dataset = train_image_path_dataset.map(load_image)
train_labels = tf.cast(train_labels, tf.int64)
train_label_dataset = tf.data.Dataset.from_tensor_slices(train_labels)
train_dataset = tf.data.Dataset.zip((train_image_dataset, train_label_dataset))
print(train_dataset)
train_dataset = train_dataset.shuffle(2000).repeat().batch(32)

test_image_path_dataset = tf.data.Dataset.from_tensor_slices(test_image_paths)
test_image_dataset = test_image_path_dataset.map(load_image)
test_labels = tf.cast(test_labels, tf.int64)
test_label_dataset = tf.data.Dataset.from_tensor_slices(test_labels)
test_dataset = tf.data.Dataset.zip((test_image_dataset, test_label_dataset))
print(test_dataset)
test_dataset = test_dataset.repeat().batch(32)


conv_base = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=(256, 256, 3))
conv_base.trainable = False
print(conv_base.summary())
model = tf.keras.Sequential([conv_base,
                             tf.keras.layers.GlobalAveragePooling2D(),
                             tf.keras.layers.Dense(256, activation='relu'),
                             tf.keras.layers.Dense(1, activation='softmax')])
print(model.summary())
model.compile(optimizer='adam', metrics=['accuracy'], loss='binary_crossentropy')

model.fit(train_dataset, validation_data=test_dataset, epochs=10, steps_per_epoch=2000//32, validation_steps=1000//32)

model.save('./model/dc_vgg16.h5')

# 预训练微调 需要先训练好分类器，再解冻顶层卷积继续训练
# 只有分类器已经训练好了，才能微调卷积基的顶部卷积层。如果有没有这样的话，刚开始的训练误差很大，微调之前这些卷积层学到的表示会被破坏掉
conv_base.trainable = True
fine_tune = -3
for layer in conv_base.layers[:fine_tune]:
    layer.trainable = False
print(model.summary())
model.compile(optimizer='adam', metrics=['accuracy'], loss='binary_crossentropy')
initial_epoch = 10
fine_tune_epoch = 10
total_epoch = initial_epoch+fine_tune_epoch
model.fit(train_dataset, validation_data=test_dataset, initial_epoch=initial_epoch,  epochs=total_epoch,
          steps_per_epoch=2000//32, validation_steps=1000//32)



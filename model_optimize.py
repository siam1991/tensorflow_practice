import tensorflow as tf
import matplotlib.pyplot as plt

(train_img, train_label), (test_img, test_label) = tf.keras.datasets.fashion_mnist.load_data()
print(train_img.shape)
print(train_label.shape)

print(train_img[0])
train_img = train_img/255
test_img = test_img/255

# 1. 增加模型深度
# 2. 出现过拟合，使用dropout层
# 3. 测试集准确率>训练集准确率
"""
过拟合：增大数据集, 降低模型复杂度(dropout，正则化)
欠拟合: 增大模型复杂度（模型深度，神经元个数）
3个正则化关键字
1. kernel_regularizer：对权值进行正则化，大多数情况下使用这个
2. bias_regularizer：限制bias的大小，使得输入和输出接近
3. activity_regularizer：对输出进行正则化，使得输出尽量小
"""
model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)),
                             tf.keras.layers.Dense(128, activation='relu'),
                             tf.keras.layers.Dense(128, activation='relu'),  # 3
                             tf.keras.layers.Dropout(0.5),  # 2
                             tf.keras.layers.Dense(128, activation='relu'),
                             tf.keras.layers.Dense(128, activation='relu'),  # 3
                             tf.keras.layers.Dropout(0.5),  # 2
                             tf.keras.layers.Dense(128, activation='relu'),
                             tf.keras.layers.Dense(128, activation='relu'),  # 3
                             tf.keras.layers.Dropout(0.5),  # 2
                             tf.keras.layers.Dense(10, activation='softmax')])
print(model.summary())
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_img, train_label, epochs=30, validation_data=(test_img, test_label))

plt.plot(history.epoch, history.history['loss'], color='blue', label='loss')
plt.plot(history.epoch, history.history['val_loss'], color='red', label='var_loss')
plt.legend()
plt.show()

plt.plot(history.epoch, history.history['accuracy'], color='blue', label='accuracy')
plt.plot(history.epoch, history.history['val_accuracy'], color='red', label='val_accuracy')
plt.legend()
plt.show()




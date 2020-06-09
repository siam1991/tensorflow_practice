import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
"""
MLP即多层感知机，其实就是多层神经网络,为了拟合非线性，引入了激活函数的概念
激活函数： relu, leak relu tanh sigmoid,一般常用的激活函数为relu
"""
data = pd.read_csv('./data/advertising.csv')
print(data.head())
x = data.iloc[:, 1:-1]
y = data.iloc[:, -1]
# plt.scatter(data.TV, data.sales,color='red')
# plt.scatter(data.radio, data.sales, color='green')
# plt.scatter(data.newspaper, data.sales, color='blue')
# plt.show()

model = tf.keras.Sequential([tf.keras.layers.Dense(10, input_shape=(3,), activation='relu'),
                             tf.keras.layers.Dense(1)])
model.summary()
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
history = model.fit(x, y, epochs=1000)
model.save('./model/advertising.h5')
model.save_weights('./weights/advertising/advertising')
test_x = data.iloc[:10, 1:-1]
predict_y = model.predict(test_x)
print(predict_y)

print(history.history.keys())
# plt.plot(history.epoch, history.history['loss'])
# plt.show()

"""函数式api"""
input = tf.keras.Input(shape=(3, ))
x0 = tf.keras.layers.Dense(10, activation='relu')(input)
output = tf.keras.layers.Dense(1)(x0)
model2 = tf.keras.Model(inputs=input, outputs=output)
print(model2.summary())
model2.compile(optimizer='adam', loss='mse', metrics=['mae'])
model2.fit(x, y, epochs=1000)




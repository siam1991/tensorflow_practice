import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
print(tf.__version__)
"""
tf.keras.Sequential() == tf.kears.models.Sequential()=
tf.keras.layers.Input() == tf.keras.Input()
tf.keras.layers.Maxpooling2D() == Maxpool2D()
tf.keras.model.Model == tf.keras.Model
"""
data = pd.read_csv('data/income.csv')
x = data.Education
y = data.Income
plt.scatter(x, y)
plt.show()
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1, input_shape=(1,)))
print(model.summary())
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(x, y, epochs=10000)
# 保存模型，保存权重
model.save('./model/income.h5')
model.save_weights('./weights/income/income')
y_predict = model.predict(x)
plt.scatter(x, y, color='green')
plt.scatter(x, y_predict, color='red')
plt.show()

# 加载模型
model = tf.keras.models.load_model('./model/income.h5')
predict_sample = model.predict(pd.Series([17.5]))
print(predict_sample)

# 加载权重，首先需要重建构建模型
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1, input_shape=(1,)))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.load_weights('./weights/income/income')
predict_sample = model.predict(pd.Series([17.5]))
print(predict_sample)

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
"""
逻辑回归是分类模型，使用sigmoid作为激活函数时，使用交叉熵作为损失函数，
之前的均方差也可作为loss,但使用二元交叉熵更好，可以放大差异，使得loss与预测值在同一数量级上
交叉熵公式 -[plog(q)+(1-p)log(1-q)], q位sigmoid函数输出，永远不会等于0或1，只会无限趋近于0或1

目标函数(obj function)，损失函数（loss function）,代价函数（cost function）
损失函数和代价函数是同一个东西 loss==cost
目标函数是一个与他们相关但更广的概念，
对于目标函数来说在有约束条件下的最小化就是损失函数（loss function）

目标函数是最终需要优化的函数，其中包括经验损失和结构损失。
obj=loss+Ω
经验损失(loss)就是传说中的损失函数或者代价函数。结构损失(Ω)就是正则项之类的来控制模型复杂程度的函数。
"""

data = pd.read_csv('./data/credit-a.csv', header=None)
print(data.head())
x = data.iloc[:, :-1]
y = data.iloc[:, -1].replace(-1, 0)
print(y.value_counts())

model = tf.keras.Sequential([tf.keras.layers.Dense(4, input_shape=(15, ), activation='relu'),
                             tf.keras.layers.Dense(4, activation='relu'),
                             tf.keras.layers.Dense(1, activation='sigmoid')])
print(model.summary())
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(x, y, epochs=500)
model.save('./model/credit.h5')
model.save_weights('./weights/credit/credit')

print(history.history.keys())
plt.plot(history.epoch, history.history['loss'])
plt.plot(history.epoch, history.history['accuracy'])
plt.show()

"""函数式api"""
input = tf.keras.Input(shape=(15, ))
x0 = tf.keras.layers.Dense(4, activation='relu')(input)
x1 = tf.keras.layers.Dense(4, activation='relu')(x0)
output = tf.keras.layers.Dense(1, activation='sigmoid')(x1)
model2 = tf.keras.Model(inputs=input, outputs=output)
print(model2.summary())
model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model2.fit(x, y, epochs=500)


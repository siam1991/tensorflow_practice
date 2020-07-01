import tensorflow as tf

tf.executing_eagerly()

x = [[2, ]]
m = tf.matmul(x, x)
print(m)
print(m.numpy())

a = tf.constant([[1, 2],
                 [3, 4]])
print(a)
print(a.numpy())

b = tf.add(a, 1)
print(b)

print((a+b).numpy())

g = tf.convert_to_tensor(10)
print(g)
print(float(g))

c = tf.multiply(a, b)
print(c)

num = tf.convert_to_tensor(10)
print(num)
for i in range(num.numpy()):
    i = tf.constant(i)
    if int(i % 2) == 0:
        print('even')
    else:
        print('odd')

v = tf.Variable(0.0)
print((v+1).numpy())

v.assign(5)
print(v)

v.assign_add(1)
print(v)

v.read_value()

w = tf.Variable([[1.0]])
with tf.GradientTape() as t:
    loss = w*w
grad = t.gradient(loss, w)
print(grad)

w = tf.constant(3.0)
with tf.GradientTape() as t:
    t.watch(w)
    loss = w*w
dloss_dw = t.gradient(loss, w)
print(dloss_dw)

w = tf.constant(3.0)
with tf.GradientTape(persistent=True) as t:
    t.watch(w)
    y = w*w
    z = y*y
dy_dw = t.gradient(y, w)
dz_dw = t.gradient(z, w)
print(dy_dw)
print(dz_dw)


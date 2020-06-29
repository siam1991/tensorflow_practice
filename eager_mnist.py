import tensorflow as tf

tf.executing_eagerly()

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
print(train_images.shape)
train_images = tf.expand_dims(train_images, -1)
print(train_images.shape)
test_images = tf.expand_dims(test_images, -1)
train_images = tf.cast(train_images/255, tf.float32)
train_labels = tf.cast(train_labels, tf.int64)
test_images = tf.cast(test_images/255, tf.float32)
test_labels = tf.cast(test_labels, tf.int64)
dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
dataset = dataset.shuffle(30000).batch(32)


test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
test_dataset = test_dataset.batch(32)

model = tf.keras.Sequential([tf.keras.layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'),
                             tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                             tf.keras.layers.GlobalMaxPooling2D(),
                             tf.keras.layers.Dense(10)
                             ])

# logits表示网络的直接输出 。没经过sigmoid或者softmax的概率化。from_logits=False就表示把已经概率化了的输出，重新映射回原值
loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean('train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')

test_loss = tf.keras.metrics.Mean('test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')


def step_train(model, train_images, train_labels):
    with tf.GradientTape() as t:
        predict = model(train_images)
        loss_step = loss_func(train_labels, predict)
    grads = t.gradient(loss_step, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_loss(loss_step)
    train_accuracy(train_labels, predict)


def step_test( model, test_images, test_labels):
    predict = model(test_images)
    loss_step = loss_func(test_labels, predict)
    test_loss(loss_step)
    test_accuracy(test_labels, predict)


def train():
    for epoch in range(10):
        for batch, (train_images, train_labels) in enumerate(dataset):
            step_train(model, train_images, train_labels)
        print("epoch:{} loss:{} accuracy:{}".format(epoch, train_loss.result(), train_accuracy.result()))
        for batch, (test_images, test_labels) in enumerate(test_dataset):
            step_test(model, test_images, test_labels)
        print("epoch:{} loss:{} accuracy:{}".format(epoch, test_loss.result(), test_accuracy.result()))
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()


if __name__ == '__main__':
    train()
    """
    epoch:0 loss:0.7459813356399536 accuracy:0.7669000029563904
    epoch:0 loss:0.3718707263469696 accuracy:0.8830999732017517
    epoch:1 loss:0.3527647852897644 accuracy:0.8863000273704529
    epoch:1 loss:0.2959372103214264 accuracy:0.9043999910354614
    epoch:2 loss:0.2920496463775635 accuracy:0.9067166447639465
    epoch:2 loss:0.2614816427230835 accuracy:0.9157999753952026
    epoch:3 loss:0.26013216376304626 accuracy:0.916783332824707
    epoch:3 loss:0.2386365830898285 accuracy:0.9241999983787537
    epoch:4 loss:0.2395881563425064 accuracy:0.923799991607666
    epoch:4 loss:0.2049219310283661 accuracy:0.9387999773025513
    epoch:5 loss:0.22177475690841675 accuracy:0.9307833313941956
    epoch:5 loss:0.20913255214691162 accuracy:0.9330000281333923
    epoch:6 loss:0.2114388793706894 accuracy:0.9330000281333923
    epoch:6 loss:0.20462960004806519 accuracy:0.9351999759674072
    epoch:7 loss:0.20102578401565552 accuracy:0.9362666606903076
    epoch:7 loss:0.18441230058670044 accuracy:0.9434000253677368
    epoch:8 loss:0.19076862931251526 accuracy:0.9389166831970215
    epoch:8 loss:0.18929797410964966 accuracy:0.9402999877929688
    epoch:9 loss:0.18570120632648468 accuracy:0.9412166476249695
    epoch:9 loss:0.18279309570789337 accuracy:0.9430999755859375
    """


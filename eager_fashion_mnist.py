import tensorflow as tf
import numpy as np
tf.executing_eagerly()

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

print(train_images.shape)
train_images = tf.expand_dims(train_images, -1)
print(train_images.shape)
test_images = tf.expand_dims(test_images, -1)
train_images = tf.cast(train_images/255, tf.float32)
test_images = tf.cast(test_images/255, tf.float32)
train_labels = tf.cast(train_labels, tf.int64)
test_labels = tf.cast(test_labels, tf.int64)

dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
test_datasetet = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

dataset = dataset.shuffle(20000).batch(32)
test_dataset = test_datasetet.batch(32)

model = tf.keras.Sequential([tf.keras.layers.Conv2D(64, (3, 3), input_shape=(28, 28, 1), activation='relu'),
                             tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                             tf.keras.layers.MaxPool2D(),
                             tf.keras.layers.Flatten(),
                             # tf.keras.layers.GlobalMaxPooling2D(),
                             tf.keras.layers.Dense(256, activation='relu'),
                             tf.keras.layers.Dense(10)
                             ])
print(model.summary())
loss_func = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.optimizers.Adam()

train_loss = tf.keras.metrics.Mean('train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
test_loss = tf.keras.metrics.Mean('test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')


def step_train(model, images, labels):
    with tf.GradientTape() as t:
        predict = model(images)
        loss_step = loss_func(labels, predict)
    grads = t.gradient(loss_step, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_loss(loss_step)
    train_accuracy(labels, predict)


def step_test(model, images, labels):
    predict = model(images)
    loss_step = loss_func(labels, predict)
    test_loss(loss_step)
    test_accuracy(labels, predict)


def train():
    for i in range(10):
        for batch, (images, labels) in enumerate(dataset):
            step_train(model, images, labels)
        print("epoch:{} train loss:{} train accuracy:{}".format(i, train_loss.result(), train_accuracy.result()))
        for batch, (images, labels) in enumerate(test_dataset):
            step_test(model, images, labels)
        print("epoch:{} test loss:{} test accuracy:{}".format(i, test_loss.result(), test_accuracy.result()))


if __name__ == '__main__':
    train()
    """
    epoch:0 train loss:0.3669969141483307 train accuracy:0.8671666383743286
    epoch:0 test loss:0.2806003987789154 test accuracy:0.895799994468689
    epoch:1 train loss:0.29861608147621155 train accuracy:0.8915249705314636
    epoch:1 test loss:0.2606083154678345 test accuracy:0.9053000211715698
    epoch:2 train loss:0.25804612040519714 train accuracy:0.9054499864578247
    epoch:2 test loss:0.24679480493068695 test accuracy:0.9108999967575073
    epoch:3 train loss:0.22701111435890198 train accuracy:0.9165208339691162
    epoch:3 test loss:0.2445085197687149 test accuracy:0.9130499958992004
    epoch:4 train loss:0.20187339186668396 train accuracy:0.92562335729599
    epoch:4 test loss:0.2438361495733261 test accuracy:0.9148799777030945
    epoch:5 train loss:0.18022358417510986 train accuracy:0.9334972500801086
    epoch:5 test loss:0.2521045506000519 test accuracy:0.9163166880607605
    epoch:6 train loss:0.16203242540359497 train accuracy:0.9402261972427368
    epoch:6 test loss:0.260092556476593 test accuracy:0.9172571301460266
    epoch:7 train loss:0.14688995480537415 train accuracy:0.9458291530609131
    epoch:7 test loss:0.27395913004875183 test accuracy:0.9177374839782715
    epoch:8 train loss:0.1342746913433075 train accuracy:0.9504963159561157
    epoch:8 test loss:0.2859102189540863 test accuracy:0.9187555313110352
    epoch:9 train loss:0.12379144877195358 train accuracy:0.9543949961662292
    epoch:9 test loss:0.2999812662601471 test accuracy:0.9189000129699707
    """
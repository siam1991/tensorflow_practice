import tensorflow as tf

imdb_data = tf.keras.datasets.imdb
(x_train, y_train), (x_test, y_test) = imdb_data.load_data(num_words=10000)
print(x_train.shape, y_train.shape)
print(len(imdb_data.get_word_index()))

x_train_length = [len(x) for x in x_train]
print(x_train_length)

x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=300)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=300)
x_train_length = [len(x) for x in x_train]
print(x_train_length)

model = tf.keras.Sequential([tf.keras.layers.Embedding(input_dim=10000, output_dim=50, input_length=300),
                             tf.keras.layers.GlobalAveragePooling1D(),
                             # tf.keras.layers.Flatten(),
                             tf.keras.layers.Dense(128, activation='relu',
                                                   kernel_regularizer=tf.keras.regularizers.l2(0.01)),
                             # tf.keras.layers.Dropout(0.5),
                             tf.keras.layers.Dense(1, activation='sigmoid')])
print(model.summary())
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=20, batch_size=256, validation_data=(x_test, y_test))
model.save('./model/imdb.h5')
model.save_weights('./weights/imdb/imdb')

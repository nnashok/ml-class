import tensorflow as tf
import wandb

# logging code
run = wandb.init()
config = run.config

# load data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

img_width = X_train.shape[1]
img_height = X_train.shape[2]

# normalize data [0-1]
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

# reshape input data
X_train = X_train.reshape(
    X_train.shape[0], img_width, img_height, 1)
X_test = X_test.reshape(
    X_test.shape[0], img_width, img_height, 1)

# one hot encode outputs
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)
labels = [str(i) for i in range(10)]

print(y_train.shape, y_test.shape)

num_classes = y_train.shape[1]

# create model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(16, (3,3), input_shape=(img_width, img_height, 1)))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(1000, activation="relu"))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(num_classes, activation="softmax"))
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test),
          callbacks=[wandb.keras.WandbCallback(data_type="image", labels=labels, save_model=False)])

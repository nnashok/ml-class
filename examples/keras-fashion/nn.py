import tensorflow as tf
import wandb

# logging code
run = wandb.init()
config = run.config
config.epochs = 100
config.lr = 0.01
config.layers = 3
config.dropout = 0.4
config.hidden_layer_1_size = 128

# load data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Normalize data
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

img_width = X_train.shape[1]
img_height = X_train.shape[2]
labels = ["T-shirt/top", "Trouser", "Pullover", "Dress",
          "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# reshape input data
X_train = X_train.reshape(
    X_train.shape[0], img_width, img_height, 1)
X_test = X_test.reshape(
    X_test.shape[0], img_width, img_height, 1)

# one hot encode outputs
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

num_classes = y_train.shape[1]

# create model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(16, (3,3), input_shape=(img_width, img_height, 1)))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Flatten(input_shape=(28,28, 1)))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(1000, activation="relu"))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(num_classes, activation="softmax"))
model.compile(loss='mse', optimizer='adam',
              metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train, epochs=config.epochs, validation_data=(X_test, y_test),
          callbacks=[wandb.keras.WandbCallback(data_type="image", labels=labels)])

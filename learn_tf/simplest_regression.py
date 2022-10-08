import tensorflow as tf

X = tf.constant([-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(X) # X's shape is (7,)
X = tf.expand_dims(X, axis=-1)
print(X) # X's shape is (7, 1)
y = tf.constant([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]) # y's shape is (7,)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1)
])

## model.build is not necessary for the training.
## It's only for the correctness of calling model.summary() 
model.build(input_shape=(None, 1))
model.summary()


model.compile(
    loss = tf.keras.losses.mae,
    optimizer = tf.keras.optimizers.SGD(),
    metrics=["mae"])

model.fit(x=X, y=y, epochs=100) # loss reduced from 8.7969 to 0.1742


print(model.predict([17])) # prediction is 18.425007 
print(model.trainable_variables) #get the model weights, 1.0451965*x + 0.6566663
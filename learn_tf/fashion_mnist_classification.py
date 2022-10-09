import tensorflow as tf
from keras.datasets import fashion_mnist

# train_data's shape is (60000, 28, 28), train_labels' shape is (60000,) 
(train_data, train_labels), (test_data, test_labels) = fashion_mnist.load_data()

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)), 
    tf.keras.layers.Dense(4, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(4, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(10, activation=tf.keras.activations.softmax)
])

model.compile(
    optimizer = tf.keras.optimizers.Adam(),
    loss = tf.keras.losses.sparse_categorical_crossentropy,
    metrics = ["accuracy"]
)

# validation_data is used to see if the model performs well on 
# the test dataset during the training 
# non_norm_history = model.fit(x=train_data, y=train_labels, epochs=10, validation_data=(test_data, test_labels))

# normalization is very important as the non_norm version only produces the model with accuracy about 0.3.
norm_history = model.fit(x=train_data/255.0, y=train_labels, epochs=10, validation_data=(test_data/255.0, test_labels))
# The accuracy on test_data is ~ 0.76

model.summary()
# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  flatten (Flatten)           (None, 784)               0         
                                                                 
#  dense (Dense)               (None, 4)                 3140      
                                                                 
#  dense_1 (Dense)             (None, 4)                 20        
                                                                 
#  dense_2 (Dense)             (None, 10)                50        
                                                                 
# =================================================================
# Total params: 3,210
# Trainable params: 3,210
# Non-trainable params: 0
# _________________________________________
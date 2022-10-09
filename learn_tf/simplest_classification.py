import tensorflow as tf
from sklearn.datasets import make_circles

# Make 1000 examples
n_samples = 1000
X, y = make_circles(n_samples=n_samples, 
                    noise=0.03, 
                    random_state=42)


model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(4, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(4, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)
])

model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
    loss = tf.keras.losses.binary_crossentropy,
    metrics = ["accuracy"]
)

model.fit(x=X, y = y, epochs=50)

model.summary()
# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  dense (Dense)               (None, 4)                 12        
                                                                 
#  dense_1 (Dense)             (None, 4)                 20        
                                                                 
#  dense_2 (Dense)             (None, 1)                 5         
                                                                 
# =================================================================
# Total params: 37
# Trainable params: 37
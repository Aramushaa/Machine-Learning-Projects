import tensorflow as tf
from tensorflow.keras import layers, models
import tqdm
import plotly.express as px

def swissRoll(samples=1000, noise=0.0):
    t = 1.5 * 3.14 * (1 + 2*tf.random.uniform((samples, 1)))
    x = t * tf.math.cos(t)
    y = t * tf.math.sin(t)
    X = tf.concat([x, y], axis=1)
    if noise > 0:
        X += noise * tf.random.normal((samples, 2))
    return X

def makeBlock(in_features, out_features):
    return models.Sequential([
        layers.Dense(out_features, input_shape=(in_features,), activation=None),
        layers.BatchNormalization(),
        layers.ReLU()
    ])

encoder = models.Sequential([
    makeBlock(2, 64),
    makeBlock(64, 64),
    makeBlock(64, 64),
    makeBlock(64, 64),
    makeBlock(64, 64),
    layers.Dense(4, activation=None)
])


decoder = models.Sequential([
    makeBlock(2, 64),
    makeBlock(64, 64),
    makeBlock(64, 64),
    makeBlock(64, 64),
    makeBlock(64, 64),
    layers.Dense(2, activation=None)
])

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

@tf.function
def sp(latent):
    mean = latent[:, :2]
    logVar = latent[:, 2:]
    std = tf.exp(0.5 * logVar)
    latentSample = mean + std * tf.random.normal(tf.shape(std))
    kl = tf.reduce_mean(0.5 * tf.reduce_sum(tf.square(mean) + tf.exp(logVar) - logVar - 1, axis=-1))
    return latentSample, kl

@tf.function
def train_step(sample):
    with tf.GradientTape() as tape:
        latent = encoder(sample)  # Pass input data through the encoder
        latentSample, klLoss = sp(latent)  # Sample from the latent space and compute KL divergence
        y = decoder(latentSample)  # Reconstruct the input
        mse = tf.keras.losses.MeanSquaredError()  # Define MSE loss
        reconstructionLoss = mse(sample, y)  # Calculate the reconstruction loss
        loss = reconstructionLoss + klLoss  # Total loss: reconstruction + KL divergence
    gradients = tape.gradient(loss, encoder.trainable_variables + decoder.trainable_variables)
    optimizer.apply_gradients(zip(gradients, encoder.trainable_variables + decoder.trainable_variables))
    return loss


loss = 0.0
# Training Loop
for i in (pbar := tqdm.tqdm(range(2000))):
    sample = swissRoll(1024, 0.5)
    loss = train_step(sample)
    if i % 100 == 0:
        pbar.set_postfix({'Loss': loss.numpy()})

# Generate data
sR = swissRoll(500, 0.5)
latent = encoder(sR)
latent, _ = sp(latent)
latent = tf.random.normal(tf.shape(latent))
generatedData = decoder(latent).numpy()

# Plot
px.scatter(x=sR[:, 0], y=sR[:, 1], width=512, height=512, template='plotly_dark', range_x=[-15, 15], range_y=[-15, 15]).show()
px.scatter(x=generatedData[:, 0], y=generatedData[:, 1], width=512, height=512, template='plotly_dark', range_x=[-15, 15], range_y=[-15, 15]).show()
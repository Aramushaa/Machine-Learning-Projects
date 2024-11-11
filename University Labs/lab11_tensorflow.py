#%%
import tensorflow as tf
from tqdm.auto import tqdm
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
    return tf.keras.models.Sequential([
        tf.keras.layers.Dense(out_features, input_shape=(in_features,), activation=None),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU()
    ])

def get_coefficients(t):
    alpha = tf.math.cos(3.14 * t / 2)
    beta = tf.math.sin(3.14 * t / 2)
    return alpha, beta

decoder = tf.keras.models.Sequential([
    makeBlock(2, 64),
    makeBlock(64, 64),
    makeBlock(64, 64),
    makeBlock(64, 64),
    makeBlock(64, 64),
    tf.keras.layers.Dense(2, activation=None)
])

optimizerD = tf.keras.optimizers.Adam(learning_rate=1e-3)
# Training Loop
loss_fn = tf.keras.losses.MeanSquaredError()
for i in (pbar := tqdm(range(2000))):
    sample = swissRoll(1024, 0.5)

    t = tf.random.uniform((sample.shape[0], 1), dtype=tf.float32)
    alpha, beta = get_coefficients(t)
    noise = tf.random.normal(shape=sample.shape)
    latent = alpha * sample + beta * noise

    with tf.GradientTape() as tape:
        prediction = decoder(latent,training=True)
        loss = loss_fn(sample, prediction)
    
    gradients = tape.gradient(loss, decoder.trainable_variables)
    optimizerD.apply_gradients(zip(gradients, decoder.trainable_variables))
    
    if i % 100 == 0:
        pbar.set_postfix({'Loss': loss.numpy()})

#%%
# Generate Data
sR = swissRoll(500, 0.5)
latent = tf.random.normal(shape=sR.shape)
t = 1.0
for i in range(20):
    alpha, beta = get_coefficients(t)
    latent = decoder(latent,training=False)
    t -= 0.05
    latent = alpha * latent + beta * tf.random.normal(shape=latent.shape)

generatedData = latent.numpy()

px.scatter(x=sR[:,0], y=sR[:,1], width=512, height=512, template='plotly_dark', range_x=[-15, 15], range_y=[-15, 15]).show()
px.scatter(x=generatedData[:,0], y=generatedData[:,1], width=512, height=512, template='plotly_dark', range_x=[-15, 15], range_y=[-15, 15]).show()
# %%

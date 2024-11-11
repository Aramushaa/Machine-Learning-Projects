import tensorflow as tf
from tqdm.auto import tqdm
from plotly import express as px

def swissRoll(samples=1000, noise=0.0):
    t = 1.5 * 3.14 * (1 + 2*tf.random.uniform((samples, 1)))
    x = t * tf.math.cos(t)
    y = t * tf.math.sin(t)
    X = tf.concat([x, y], axis=1)
    if noise > 0:
        X += noise * tf.random.normal((samples, 2))
    return X

def makeBlock(in_features, out_features, bn=True):
    block = tf.keras.Sequential()
    block.add(tf.keras.layers.Dense(out_features, input_shape=(in_features,)))
    if bn:
        block.add(tf.keras.layers.BatchNormalization())
    block.add(tf.keras.layers.ReLU())
    return block

def hinge(x, maximize=True):
    if maximize:
        return tf.nn.relu(1 - x)
    else:
        return tf.nn.relu(1 + x)
    
encoder = tf.keras.Sequential([
    makeBlock(2, 64, bn=False),
    makeBlock(64, 64),
    makeBlock(64, 64),
    makeBlock(64, 64),
    makeBlock(64, 64),
    tf.keras.layers.Dense(1)
])

decoder = tf.keras.Sequential([
    makeBlock(2, 64),
    makeBlock(64, 64),
    makeBlock(64, 64),
    makeBlock(64, 64),
    makeBlock(64, 64),
    tf.keras.layers.Dense(2)
])

optimizerD = tf.keras.optimizers.Adam(learning_rate=5e-4)
optimizerE = tf.keras.optimizers.Adam(learning_rate=1e-3)

@tf.function
def train_step(sample):
    with tf.GradientTape() as tapeD, tf.GradientTape() as tapeE:
        latent = tf.random.normal(sample.shape)
        sampleFake = decoder(latent)
        generatorLoss = tf.reduce_mean(hinge(encoder(sampleFake), maximize=True))
        
        discriminatorLoss = tf.reduce_mean(
            hinge(encoder(sampleFake), maximize=False) + 
            hinge(encoder(sample), maximize=True)
        )

    gradientsD = tapeD.gradient(generatorLoss, decoder.trainable_variables)
    optimizerD.apply_gradients(zip(gradientsD, decoder.trainable_variables))
    
    gradientsE = tapeE.gradient(discriminatorLoss, encoder.trainable_variables)
    optimizerE.apply_gradients(zip(gradientsE, encoder.trainable_variables))
    
    return generatorLoss,discriminatorLoss

# Training Loop
for i in (pbar:=tqdm(range(8000))):
    sample = swissRoll(1024, 0.5)
    generatorLoss,discriminatorLoss = train_step(sample)
    if i % 100 == 0:
        pbar.set_postfix({'Generator Loss': generatorLoss.numpy()})

sR = swissRoll(500, 0.5)
latent = tf.random.normal(sR.shape)
generatedData = decoder(latent, training=False)

px.scatter(x=sR[:, 0], y=sR[:, 1], width=512, height=512, template='plotly_dark', range_x=[-15, 15], range_y=[-15, 15]).show()
px.scatter(x=generatedData[:, 0], y=generatedData[:, 1], width=512, height=512, template='plotly_dark', range_x=[-15, 15], range_y=[-15, 15]).show()
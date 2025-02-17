import tensorflow as tf
import preprocessing, callbacks
from tensorflow import ResNet50, Dense, GlobalAveragePooling2D, Dropout, Model

# Using ResNet50 as pretrained model to transfer learning to (exclude top layers)
base_model = ResNet50(
    weights = 'imagenet', 
    include_top = False, 
    input_shape = (224, 224, 3)  # matches resized images
)

for layer in base_model.layers[:15]:
    layer.trainable = False

# saves output from base model
output = base_model.output
# output is converted into 1D vector to average out the base output
output = GlobalAveragePooling2D()(output)
# output is condensed to help model learn relationships between highlevel features
output = Dense(512, activation = 'relu')(output)
# randomly drops 50% of neurons in output
# allows model to get better at generalizing by reducing reliance on specific neurons
output = Dropout(0.5)(output)
# finally, the output is transformed into probablities for each lung diagnoses type (COVID, Lung Opacity, Viral Pneumonia, and Normal)
predictions = Dense(4, activation = 'softmax')(output)

# now we compile the model
model = Model(inputs = base_model.inputs, outputs=predictions)

model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4), # or 0.0001, rate commonly used for transfer learning
    loss = 'categorical_crossentropy',  # for multi-class classification
    metrics = [
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

# model history saves the training history of the model as it learns
model_history = model.fit(
    preprocessing.train_generator,
    epochs = 30, # one epoch is one full pass through the training data
    validation_data = preprocessing.val_generator,
    callbacks = callbacks.callbacks,
)

# fine tuning

for layer in base_model.layers[15:]:
    layer.trainable = True

# recompiling model using smaller learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    preprocessing.train_generator, 
    epochs = 10, 
    validation_data = preprocessing.val_generator
    )
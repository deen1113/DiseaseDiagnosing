import tensorflow as tf
from testing import test_model
from matrix import conf_matrix
import preprocessing

# loads best model from training
model = tf.keras.models.load_model('best_model.h5', compile = False)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# runs tests
test_model(model)
conf_matrix(model)
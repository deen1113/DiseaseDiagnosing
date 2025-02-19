from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

'''
Callbacks are used to improve training stability and performance.
Here I use three, EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint.
EarlyStopping - Stops training the model early if it stops improving.
ReduceLROnPlateau - Reduces learning rate when progress slows down.
ModelCheckpoint - Saves the best model during training, even if later models overfit.
'''

callbacks = [
    EarlyStopping(patience = 5, restore_best_weights = True),
    ReduceLROnPlateau(factor = 0.1, patience = 2),
    ModelCheckpoint('best_model.h5', save_best_only = True)
]
from utils import *
import os
from keras.callbacks import *
from keras.optimizers import *

MODEL_PATH = './model/model_{val_loss:.5f}.h5'
if not os.path.exists('./model'):
    os.mkdir('./model')

model = build_model()
optimizer = Adam(lr=0.0001)
model.compile(optimizer=optimizer, loss={'decoded_image': 'mse', 'container_image': 'mse'}, loss_weights={
              'decoded_image': 1., 'container_image': 1.})


train_image_list = os.listdir(TRAIN_FOLDER)
train_image_list = [os.path.join(TRAIN_FOLDER, i) for i in train_image_list]
validate_image_list = os.listdir(VALIDATE_FOLDER)
validate_image_list = [os.path.join(VALIDATE_FOLDER, i) for i in validate_image_list]

check_point = ModelCheckpoint(MODEL_PATH, verbose=True, save_best_only=True)
early_stop = EarlyStopping(patience=5, verbose=True)

model.fit_generator(generate_data(train_image_list),
                    steps_per_epoch=100, epochs=500, validation_data=generate_data(validate_image_list),
                    validation_steps=100, callbacks=[check_point, early_stop])

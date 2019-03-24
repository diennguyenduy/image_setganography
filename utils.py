from keras.models import *
from keras.layers import *
import cv2
import random
import os

IMAGE_SIZE = 256
IMAGE_CHANNEL = 3
BATCH_SIZE = 8
TRAIN_FOLDER = 'train_image'
VALIDATE_FOLDER = 'validate_image'


def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    return image / 255.

def generate_data(image_list):
    while True:
        image_1_batch = []
        image_2_batch = []
        for ind in range(BATCH_SIZE):
            image_1 = load_image(random.choice(image_list))
            image_2 = load_image(random.choice(image_list))
            image_1_batch.append(image_1)
            image_2_batch.append(image_2)
        yield ({'secret_image': np.array(image_1_batch), 'cover_image': np.array(image_2_batch)},
            {'decoded_image': np.array(image_1_batch), 'container_image': np.array(image_2_batch)})

def build_model():
    secret_iamge = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL), name="secret_image")
    cover_image = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL), name="cover_image")

    encoded_image_3 = Conv2D(50, activation='relu', kernel_size=(
        3, 3), padding="same")(secret_iamge)
    encoded_image_3 = Conv2D(50, activation='relu', kernel_size=(
        3, 3), padding="same")(encoded_image_3)
    encoded_image_3 = Conv2D(50, activation='relu', kernel_size=(
        3, 3), padding="same")(encoded_image_3)
    encoded_image_3 = Conv2D(50, activation='relu', kernel_size=(
        3, 3), padding="same")(encoded_image_3)
    encoded_image_4 = Conv2D(50, activation='relu', kernel_size=(
        4, 4), padding="same")(secret_iamge)
    encoded_image_4 = Conv2D(50, activation='relu', kernel_size=(
        4, 4), padding="same")(encoded_image_4)
    encoded_image_4 = Conv2D(50, activation='relu', kernel_size=(
        4, 4), padding="same")(encoded_image_4)
    encoded_image_4 = Conv2D(50, activation='relu', kernel_size=(
        4, 4), padding="same")(encoded_image_4)
    encoded_image_5 = Conv2D(50, activation='relu', kernel_size=(
        5, 5), padding="same")(secret_iamge)
    encoded_image_5 = Conv2D(50, activation='relu', kernel_size=(
        5, 5), padding="same")(encoded_image_5)
    encoded_image_5 = Conv2D(50, activation='relu', kernel_size=(
        5, 5), padding="same")(encoded_image_5)
    encoded_image_5 = Conv2D(50, activation='relu', kernel_size=(
        5, 5), padding="same")(encoded_image_5)
    encoded_image = Concatenate(axis=-1)(
        [encoded_image_3, encoded_image_4, encoded_image_5])
    encoded_image_3 = Conv2D(50, activation='relu', kernel_size=(
        3, 3), padding="same")(encoded_image)
    encoded_image_4 = Conv2D(50, activation='relu', kernel_size=(
        4, 4), padding="same")(encoded_image)
    encoded_image_5 = Conv2D(50, activation='relu', kernel_size=(
        5, 5), padding="same")(encoded_image)
    encoded_image = Concatenate(axis=-1)(
        [encoded_image_3, encoded_image_4, encoded_image_5])
    concat_input = Concatenate(axis=-1)([
        encoded_image, cover_image])
    decoded_image_3 = Conv2D(50, activation='relu', kernel_size=(
        3, 3), padding="same")(concat_input)
    decoded_image_3 = Conv2D(50, activation='relu', kernel_size=(
        3, 3), padding="same")(decoded_image_3)
    decoded_image_3 = Conv2D(50, activation='relu', kernel_size=(
        3, 3), padding="same")(decoded_image_3)
    decoded_image_3 = Conv2D(50, activation='relu', kernel_size=(
        3, 3), padding="same")(decoded_image_3)
    decoded_image_4 = Conv2D(50, activation='relu', kernel_size=(
        4, 4), padding="same")(concat_input)
    decoded_image_4 = Conv2D(50, activation='relu', kernel_size=(
        4, 4), padding="same")(decoded_image_4)
    decoded_image_4 = Conv2D(50, activation='relu', kernel_size=(
        4, 4), padding="same")(decoded_image_4)
    decoded_image_4 = Conv2D(50, activation='relu', kernel_size=(
        4, 4), padding="same")(decoded_image_4)
    decoded_image_5 = Conv2D(50, activation='relu', kernel_size=(
        5, 5), padding="same")(concat_input)
    decoded_image_5 = Conv2D(50, activation='relu', kernel_size=(
        5, 5), padding="same")(decoded_image_5)
    decoded_image_5 = Conv2D(50, activation='relu', kernel_size=(
        5, 5), padding="same")(decoded_image_5)
    decoded_image_5 = Conv2D(50, activation='relu', kernel_size=(
        5, 5), padding="same")(decoded_image_5)
    decoded_image = Concatenate(axis=3)(
        [decoded_image_3, decoded_image_4, decoded_image_5])
    decoded_image_3 = Conv2D(50, activation='relu', kernel_size=(
        3, 3), padding="same")(decoded_image)
    decoded_image_4 = Conv2D(50, activation='relu', kernel_size=(
        4, 4), padding="same")(decoded_image)
    decoded_image_5 = Conv2D(50, activation='relu', kernel_size=(
        5, 5), padding="same")(decoded_image)
    decoded_image = Concatenate(axis=3)(
        [decoded_image_3, decoded_image_4, decoded_image_5])
    # xong hiding network
    container_image = Conv2D(IMAGE_CHANNEL, activation='relu', kernel_size=1,
                             padding="same", name="container_image")(decoded_image)
    #Decoder audio
    decoded_image_3 = Conv2D(50, activation='relu', kernel_size=(
        3, 3), padding="same")(container_image)
    decoded_image_3 = Conv2D(50, activation='relu', kernel_size=(
        3, 3), padding="same")(decoded_image_3)
    decoded_image_3 = Conv2D(50, activation='relu', kernel_size=(
        3, 3), padding="same")(decoded_image_3)
    decoded_image_3 = Conv2D(50, activation='relu', kernel_size=(
        3, 3), padding="same")(decoded_image_3)
    decoded_image_4 = Conv2D(50, activation='relu', kernel_size=(
        4, 4), padding="same")(container_image)
    decoded_image_4 = Conv2D(50, activation='relu', kernel_size=(
        4, 4), padding="same")(decoded_image_4)
    decoded_image_4 = Conv2D(50, activation='relu', kernel_size=(
        4, 4), padding="same")(decoded_image_4)
    decoded_image_4 = Conv2D(50, activation='relu', kernel_size=(
        4, 4), padding="same")(decoded_image_4)
    decoded_image_5 = Conv2D(50, activation='relu', kernel_size=(
        5, 5), padding="same")(container_image)
    decoded_image_5 = Conv2D(50, activation='relu', kernel_size=(
        5, 5), padding="same")(decoded_image_5)
    decoded_image_5 = Conv2D(50, activation='relu', kernel_size=(
        5, 5), padding="same")(decoded_image_5)
    decoded_image_5 = Conv2D(50, activation='relu', kernel_size=(
        5, 5), padding="same")(decoded_image_5)
    decoded_image = Concatenate(axis=3)(
        [decoded_image_3, decoded_image_4, decoded_image_5])
    decoded_image_3 = Conv2D(50, activation='relu', kernel_size=(
        3, 3), padding="same")(decoded_image)
    decoded_image_4 = Conv2D(50, activation='relu', kernel_size=(
        4, 4), padding="same")(decoded_image)
    decoded_image_5 = Conv2D(50, activation='relu', kernel_size=(
        5, 5), padding="same")(decoded_image)
    decoded_image = Concatenate(axis=3)(
        [decoded_image_3, decoded_image_4, decoded_image_5])
    decoded_image = Conv2D(IMAGE_CHANNEL, activation='linear', kernel_size=1,
                        padding="same", name="decoded_image")(decoded_image)
    encode_decoder = Model(inputs=[secret_iamge, cover_image], outputs=[
        container_image, decoded_image])
    print (encode_decoder.summary())
    return encode_decoder

if __name__ == '__main__':
    model = build_model()
    # image_list = os.listdir(TRAIN_FOLDER)
    # image_list = [os.path.join(TRAIN_FOLDER, i) for i in image_list]
    # print (next(generate_data(image_list)))

from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, UpSampling2D, Concatenate


def encoder_down(input, filters, dropout=False):
    x = Conv2D(filters, activation='relu', padding='same', kernel_initializer='he_normal', kernel_size=(3, 3))(input)
    x = Conv2D(filters,activation='relu', padding='same', kernel_initializer='he_normal', kernel_size=(3, 3))(x)
    p = MaxPooling2D(pool_size=(2, 2))(x)

    if dropout:
        p = Dropout(0.25)(p)

    return x, p


def encoder_up(input, filter, skip):
    x = UpSampling2D(size=(2, 2))(input)
    concat = Concatenate()([x, skip])
    x = Conv2D(filter, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(concat)
    return Conv2D(filter, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(x)


def bottle_nek(filters, input):
    x = Conv2D(filters, activation='relu', padding='same', kernel_initializer='he_normal', kernel_size=(3, 3))(input)
    return Conv2D(filters,activation='relu', padding='same', kernel_initializer='he_normal', kernel_size=(3, 3))(x)

def unet():
    inputs = Input((256, 256, 3))

    cnv1, polling1 = encoder_down(inputs, 16)
    cnv2, polling2 = encoder_down(polling1, 32)
    cnv3, polling3 = encoder_down(polling2, 64)
    cnv4, polling4 = encoder_down(polling3, 128)
    cnv5, polling5 = encoder_down(polling4, 256)
    cnv6, polling6 = encoder_down(polling5, 512)

    nek = bottle_nek(512, polling6)

    up1 = encoder_up(nek, 512, cnv6)  # 8->16:128
    up2 = encoder_up(up1, 256, cnv5)  # 8->16:128
    up3 = encoder_up(up2, 128, cnv4)  # 8->16:128
    up4 = encoder_up(up3, 64, cnv3)  # 16->32:64
    up5 = encoder_up(up4, 32, cnv2)  # 32->64:32
    up6 = encoder_up(up5, 16, cnv1)  # 64->128:16

    output = Conv2D(5, (1, 1), padding='same', activation='softmax')(up6)

    return Model(inputs=inputs, outputs=output)

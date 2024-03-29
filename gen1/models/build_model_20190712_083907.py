def build_model_20190712_083907(num_channels):
    y0 = Input(shape=(32,32,3))
    y1 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y0)
    y2 = BatchNormalization()(y1)
    y3 = LeakyReLU()(y2)
    y4 = Conv2D(1*num_channels, (1,1), padding='same', use_bias=False)(y3)
    y5 = BatchNormalization()(y4)
    y6 = LeakyReLU()(y5)
    y7 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y6)
    y8 = BatchNormalization()(y7)
    y9 = LeakyReLU()(y8)
    y10 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y9)
    y11 = BatchNormalization()(y10)
    y12 = LeakyReLU()(y11)
    y13 = Conv2D(1*num_channels, (1,1), padding='same', use_bias=False)(y12)
    y14 = BatchNormalization()(y13)
    y15 = LeakyReLU()(y14)
    y16 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y15)
    y17 = BatchNormalization()(y16)
    y18 = LeakyReLU()(y17)
    y19 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y9)
    y20 = BatchNormalization()(y19)
    y21 = LeakyReLU()(y20)
    y22 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y21)
    y23 = BatchNormalization()(y22)
    y24 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y23)
    y25 = BatchNormalization()(y24)
    y26 = LeakyReLU()(y25)
    y27 = Add()([y26, y21])
    y28 = Concatenate()([y18, y27])
    y29 = Conv2D(4*num_channels, (3,3), (2,2), padding='same', use_bias=False)(y28)
    y30 = BatchNormalization()(y29)
    y31 = LeakyReLU()(y30)
    y32 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y31)
    y33 = BatchNormalization()(y32)
    y34 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y33)
    y35 = BatchNormalization()(y34)
    y36 = LeakyReLU()(y35)
    y37 = Conv2D(4*num_channels, (1,1), padding='same', use_bias=False)(y36)
    y38 = BatchNormalization()(y37)
    y39 = LeakyReLU()(y38)
    y40 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y39)
    y41 = BatchNormalization()(y40)
    y42 = LeakyReLU()(y41)
    y43 = Concatenate()([y36, y42])
    y44 = MaxPooling2D(2,2,padding='same')(y43)
    y45 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y44)
    y46 = BatchNormalization()(y45)
    y47 = LeakyReLU()(y46)
    y48 = Conv2D(16*num_channels, (1,1), padding='same', use_bias=False)(y47)
    y49 = BatchNormalization()(y48)
    y50 = LeakyReLU()(y49)
    y51 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y50)
    y52 = BatchNormalization()(y51)
    y53 = LeakyReLU()(y52)
    y54 = MaxPooling2D(2,2,padding='same')(y53)
    y55 = Conv2D(32*num_channels, (3,3), padding='same', use_bias=False)(y54)
    y56 = BatchNormalization()(y55)
    y57 = Conv2D(32*num_channels, (3,3), padding='same', use_bias=False)(y56)
    y58 = BatchNormalization()(y57)
    y59 = LeakyReLU()(y58)
    y60 = Conv2D(32*num_channels, (3,3), padding='same', use_bias=False)(y59)
    y61 = BatchNormalization()(y60)
    y62 = Conv2D(32*num_channels, (3,3), padding='same', use_bias=False)(y61)
    y63 = BatchNormalization()(y62)
    y64 = LeakyReLU()(y63)
    y65 = Conv2D(32*num_channels, (3,3), padding='same', use_bias=False)(y59)
    y66 = BatchNormalization()(y65)
    y67 = LeakyReLU()(y66)
    y68 = Concatenate()([y64, y67])
    y69 = GlobalAveragePooling2D()(y68)
    y70 = Flatten()(y69)
    y71 = Dense(1024, use_bias=False)(y70)
    y72 = BatchNormalization()(y71)
    y73 = LeakyReLU()(y72)
    y74 = Dense(10, activation="softmax")(y73)
    y75 =  (y74)
    return Model(inputs=y0, outputs=y75)
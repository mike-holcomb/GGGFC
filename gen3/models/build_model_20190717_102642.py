def build_model_20190717_102642(num_channels):
    y0 = Input(shape=(32,32,3))
    y1 = Conv2D(1*num_channels, (1,1), padding='same', use_bias=False)(y0)
    y2 = BatchNormalization()(y1)
    y3 = LeakyReLU()(y2)
    y4 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y3)
    y5 = BatchNormalization()(y4)
    y6 = LeakyReLU()(y5)
    y7 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y6)
    y8 = BatchNormalization()(y7)
    y9 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y8)
    y10 = BatchNormalization()(y9)
    y11 = LeakyReLU()(y10)
    y12 = Conv2D(1*num_channels, (1,1), padding='same', use_bias=False)(y11)
    y13 = BatchNormalization()(y12)
    y14 = LeakyReLU()(y13)
    y15 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y14)
    y16 = BatchNormalization()(y15)
    y17 = LeakyReLU()(y16)
    y18 = Add()([y17, y11])
    y19 = Conv2D(1*num_channels, (1,1), padding='same', use_bias=False)(y18)
    y20 = BatchNormalization()(y19)
    y21 = LeakyReLU()(y20)
    y22 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y21)
    y23 = BatchNormalization()(y22)
    y24 = LeakyReLU()(y23)
    y25 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y24)
    y26 = BatchNormalization()(y25)
    y27 = LeakyReLU()(y26)
    y28 = Conv2D(1*num_channels, (1,1), padding='same', use_bias=False)(y27)
    y29 = BatchNormalization()(y28)
    y30 = LeakyReLU()(y29)
    y31 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y30)
    y32 = BatchNormalization()(y31)
    y33 = LeakyReLU()(y32)
    y34 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y33)
    y35 = BatchNormalization()(y34)
    y36 = LeakyReLU()(y35)
    y37 = Add()([y36, y27])
    y38 = Concatenate()([y37, y18])
    y39 = MaxPooling2D(2,2,padding='same')(y38)
    y40 = Conv2D(4*num_channels, (1,1), padding='same', use_bias=False)(y39)
    y41 = BatchNormalization()(y40)
    y42 = LeakyReLU()(y41)
    y43 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y42)
    y44 = BatchNormalization()(y43)
    y45 = LeakyReLU()(y44)
    y46 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y45)
    y47 = BatchNormalization()(y46)
    y48 = LeakyReLU()(y47)
    y49 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y48)
    y50 = BatchNormalization()(y49)
    y51 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y50)
    y52 = BatchNormalization()(y51)
    y53 = LeakyReLU()(y52)
    y54 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y53)
    y55 = BatchNormalization()(y54)
    y56 = LeakyReLU()(y55)
    y57 = Add()([y53, y56])
    y58 = GlobalAveragePooling2D()(y57)
    y59 = Flatten()(y58)
    y60 = Dense(1024, use_bias=False)(y59)
    y61 = BatchNormalization()(y60)
    y62 = LeakyReLU()(y61)
    y63 = Dense(10, activation="softmax")(y62)
    y64 =  (y63)
    return Model(inputs=y0, outputs=y64)
def build_model_20190717_000210(num_channels):
    y0 = Input(shape=(32,32,3))
    y1 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y0)
    y2 = BatchNormalization()(y1)
    y3 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y2)
    y4 = BatchNormalization()(y3)
    y5 = LeakyReLU()(y4)
    y6 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y5)
    y7 = BatchNormalization()(y6)
    y8 = LeakyReLU()(y7)
    y9 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y8)
    y10 = BatchNormalization()(y9)
    y11 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y10)
    y12 = BatchNormalization()(y11)
    y13 = LeakyReLU()(y12)
    y14 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y13)
    y15 = BatchNormalization()(y14)
    y16 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y15)
    y17 = BatchNormalization()(y16)
    y18 = LeakyReLU()(y17)
    y19 = Add()([y13, y18])
    y20 = MaxPooling2D(2,2,padding='same')(y19)
    y21 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y20)
    y22 = BatchNormalization()(y21)
    y23 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y22)
    y24 = BatchNormalization()(y23)
    y25 = LeakyReLU()(y24)
    y26 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y25)
    y27 = BatchNormalization()(y26)
    y28 = LeakyReLU()(y27)
    y29 = Conv2D(2*num_channels, (1,1), padding='same', use_bias=False)(y28)
    y30 = BatchNormalization()(y29)
    y31 = LeakyReLU()(y30)
    y32 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y31)
    y33 = BatchNormalization()(y32)
    y34 = LeakyReLU()(y33)
    y35 = Conv2D(2*num_channels, (1,1), padding='same', use_bias=False)(y34)
    y36 = BatchNormalization()(y35)
    y37 = LeakyReLU()(y36)
    y38 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y37)
    y39 = BatchNormalization()(y38)
    y40 = LeakyReLU()(y39)
    y41 = Add()([y40, y34])
    y42 = Concatenate()([y41, y28])
    y43 = MaxPooling2D(2,2,padding='same')(y42)
    y44 = Conv2D(8*num_channels, (1,1), padding='same', use_bias=False)(y43)
    y45 = BatchNormalization()(y44)
    y46 = LeakyReLU()(y45)
    y47 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y46)
    y48 = BatchNormalization()(y47)
    y49 = LeakyReLU()(y48)
    y50 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y49)
    y51 = BatchNormalization()(y50)
    y52 = LeakyReLU()(y51)
    y53 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y52)
    y54 = BatchNormalization()(y53)
    y55 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y54)
    y56 = BatchNormalization()(y55)
    y57 = LeakyReLU()(y56)
    y58 = Concatenate()([y49, y57])
    y59 = GlobalAveragePooling2D()(y58)
    y60 = Flatten()(y59)
    y61 = Dense(1024, use_bias=False)(y60)
    y62 = BatchNormalization()(y61)
    y63 = LeakyReLU()(y62)
    y64 = Dense(10, activation="softmax")(y63)
    y65 =  (y64)
    return Model(inputs=y0, outputs=y65)
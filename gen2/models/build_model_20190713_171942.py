def build_model_20190713_171942(num_channels):
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
    y13 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y12)
    y14 = BatchNormalization()(y13)
    y15 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y14)
    y16 = BatchNormalization()(y15)
    y17 = LeakyReLU()(y16)
    y18 = Conv2D(1*num_channels, (1,1), padding='same', use_bias=False)(y17)
    y19 = BatchNormalization()(y18)
    y20 = LeakyReLU()(y19)
    y21 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y20)
    y22 = BatchNormalization()(y21)
    y23 = LeakyReLU()(y22)
    y24 = Add()([y23, y17])
    y25 = Conv2D(1*num_channels, (1,1), padding='same', use_bias=False)(y24)
    y26 = BatchNormalization()(y25)
    y27 = LeakyReLU()(y26)
    y28 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y27)
    y29 = BatchNormalization()(y28)
    y30 = LeakyReLU()(y29)
    y31 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y30)
    y32 = BatchNormalization()(y31)
    y33 = LeakyReLU()(y32)
    y34 = Add()([y33, y30])
    y35 = Add()([y24, y34])
    y36 = Concatenate()([y35, y12])
    y37 = Conv2D(4*num_channels, (3,3), (2,2), padding='same', use_bias=False)(y36)
    y38 = BatchNormalization()(y37)
    y39 = LeakyReLU()(y38)
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
    y57 = Add()([y56, y53])
    y58 = Concatenate()([y48, y57])
    y59 = GlobalAveragePooling2D()(y58)
    y60 = Flatten()(y59)
    y61 = Dense(1024, use_bias=False)(y60)
    y62 = BatchNormalization()(y61)
    y63 = LeakyReLU()(y62)
    y64 = Dense(10, activation="softmax")(y63)
    y65 =  (y64)
    return Model(inputs=y0, outputs=y65)
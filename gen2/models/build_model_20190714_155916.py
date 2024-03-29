def build_model_20190714_155916(num_channels):
    y0 = Input(shape=(32,32,3))
    y1 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y0)
    y2 = BatchNormalization()(y1)
    y3 = LeakyReLU()(y2)
    y4 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y3)
    y5 = BatchNormalization()(y4)
    y6 = LeakyReLU()(y5)
    y7 = Concatenate()([y3, y6])
    y8 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y7)
    y9 = BatchNormalization()(y8)
    y10 = LeakyReLU()(y9)
    y11 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y10)
    y12 = BatchNormalization()(y11)
    y13 = LeakyReLU()(y12)
    y14 = Concatenate()([y7, y13])
    y15 = MaxPooling2D(2,2,padding='same')(y14)
    y16 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y15)
    y17 = BatchNormalization()(y16)
    y18 = LeakyReLU()(y17)
    y19 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y18)
    y20 = BatchNormalization()(y19)
    y21 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y20)
    y22 = BatchNormalization()(y21)
    y23 = LeakyReLU()(y22)
    y24 = Concatenate()([y23, y18])
    y25 = Conv2D(16*num_channels, (1,1), padding='same', use_bias=False)(y24)
    y26 = BatchNormalization()(y25)
    y27 = LeakyReLU()(y26)
    y28 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y27)
    y29 = BatchNormalization()(y28)
    y30 = LeakyReLU()(y29)
    y31 = Conv2D(16*num_channels, (1,1), padding='same', use_bias=False)(y30)
    y32 = BatchNormalization()(y31)
    y33 = LeakyReLU()(y32)
    y34 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y33)
    y35 = BatchNormalization()(y34)
    y36 = LeakyReLU()(y35)
    y37 = Conv2D(16*num_channels, (1,1), padding='same', use_bias=False)(y24)
    y38 = BatchNormalization()(y37)
    y39 = LeakyReLU()(y38)
    y40 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y39)
    y41 = BatchNormalization()(y40)
    y42 = LeakyReLU()(y41)
    y43 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y42)
    y44 = BatchNormalization()(y43)
    y45 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y44)
    y46 = BatchNormalization()(y45)
    y47 = LeakyReLU()(y46)
    y48 = Concatenate()([y36, y47])
    y49 = MaxPooling2D(2,2,padding='same')(y48)
    y50 = Conv2D(64*num_channels, (1,1), padding='same', use_bias=False)(y49)
    y51 = BatchNormalization()(y50)
    y52 = LeakyReLU()(y51)
    y53 = Conv2D(64*num_channels, (3,3), padding='same', use_bias=False)(y52)
    y54 = BatchNormalization()(y53)
    y55 = LeakyReLU()(y54)
    y56 = Conv2D(64*num_channels, (1,1), padding='same', use_bias=False)(y55)
    y57 = BatchNormalization()(y56)
    y58 = LeakyReLU()(y57)
    y59 = Conv2D(64*num_channels, (3,3), padding='same', use_bias=False)(y58)
    y60 = BatchNormalization()(y59)
    y61 = LeakyReLU()(y60)
    y62 = GlobalAveragePooling2D()(y61)
    y63 = Flatten()(y62)
    y64 = Dense(1024, use_bias=False)(y63)
    y65 = BatchNormalization()(y64)
    y66 = LeakyReLU()(y65)
    y67 = Dense(10, activation="softmax")(y66)
    y68 =  (y67)
    return Model(inputs=y0, outputs=y68)
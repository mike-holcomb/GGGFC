def build_model_20190712_105003(num_channels):
    y0 = Input(shape=(32,32,3))
    y1 = Conv2D(1*num_channels, (1,1), padding='same', use_bias=False)(y0)
    y2 = BatchNormalization()(y1)
    y3 = LeakyReLU()(y2)
    y4 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y3)
    y5 = BatchNormalization()(y4)
    y6 = LeakyReLU()(y5)
    y7 = Conv2D(1*num_channels, (1,1), padding='same', use_bias=False)(y6)
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
    y24 = Concatenate()([y23, y12])
    y25 = Conv2D(4*num_channels, (3,3), (2,2), padding='same', use_bias=False)(y24)
    y26 = BatchNormalization()(y25)
    y27 = LeakyReLU()(y26)
    y28 = Conv2D(4*num_channels, (1,1), padding='same', use_bias=False)(y27)
    y29 = BatchNormalization()(y28)
    y30 = LeakyReLU()(y29)
    y31 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y30)
    y32 = BatchNormalization()(y31)
    y33 = LeakyReLU()(y32)
    y34 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y33)
    y35 = BatchNormalization()(y34)
    y36 = LeakyReLU()(y35)
    y37 = Conv2D(4*num_channels, (1,1), padding='same', use_bias=False)(y36)
    y38 = BatchNormalization()(y37)
    y39 = LeakyReLU()(y38)
    y40 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y39)
    y41 = BatchNormalization()(y40)
    y42 = LeakyReLU()(y41)
    y43 = Concatenate()([y42, y33])
    y44 = Conv2D(16*num_channels, (3,3), (2,2), padding='same', use_bias=False)(y43)
    y45 = BatchNormalization()(y44)
    y46 = LeakyReLU()(y45)
    y47 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y46)
    y48 = BatchNormalization()(y47)
    y49 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y48)
    y50 = BatchNormalization()(y49)
    y51 = LeakyReLU()(y50)
    y52 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y51)
    y53 = BatchNormalization()(y52)
    y54 = LeakyReLU()(y53)
    y55 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y51)
    y56 = BatchNormalization()(y55)
    y57 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y56)
    y58 = BatchNormalization()(y57)
    y59 = LeakyReLU()(y58)
    y60 = Concatenate()([y54, y59])
    y61 = MaxPooling2D(2,2,padding='same')(y60)
    y62 = Conv2D(64*num_channels, (1,1), padding='same', use_bias=False)(y61)
    y63 = BatchNormalization()(y62)
    y64 = LeakyReLU()(y63)
    y65 = Conv2D(64*num_channels, (3,3), padding='same', use_bias=False)(y64)
    y66 = BatchNormalization()(y65)
    y67 = LeakyReLU()(y66)
    y68 = Conv2D(64*num_channels, (3,3), padding='same', use_bias=False)(y67)
    y69 = BatchNormalization()(y68)
    y70 = LeakyReLU()(y69)
    y71 = GlobalAveragePooling2D()(y70)
    y72 = Flatten()(y71)
    y73 = Dense(10, activation="softmax")(y72)
    y74 =  (y73)
    return Model(inputs=y0, outputs=y74)
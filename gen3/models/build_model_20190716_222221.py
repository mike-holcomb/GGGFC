def build_model_20190716_222221(num_channels):
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
    y13 = Concatenate()([y12, y6])
    y14 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y13)
    y15 = BatchNormalization()(y14)
    y16 = LeakyReLU()(y15)
    y17 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y16)
    y18 = BatchNormalization()(y17)
    y19 = LeakyReLU()(y18)
    y20 = Add()([y16, y19])
    y21 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y20)
    y22 = BatchNormalization()(y21)
    y23 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y22)
    y24 = BatchNormalization()(y23)
    y25 = LeakyReLU()(y24)
    y26 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y25)
    y27 = BatchNormalization()(y26)
    y28 = LeakyReLU()(y27)
    y29 = Add()([y20, y28])
    y30 = Conv2D(4*num_channels, (3,3), (2,2), padding='same', use_bias=False)(y29)
    y31 = BatchNormalization()(y30)
    y32 = LeakyReLU()(y31)
    y33 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y32)
    y34 = BatchNormalization()(y33)
    y35 = LeakyReLU()(y34)
    y36 = Conv2D(4*num_channels, (1,1), padding='same', use_bias=False)(y35)
    y37 = BatchNormalization()(y36)
    y38 = LeakyReLU()(y37)
    y39 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y38)
    y40 = BatchNormalization()(y39)
    y41 = LeakyReLU()(y40)
    y42 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y41)
    y43 = BatchNormalization()(y42)
    y44 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y43)
    y45 = BatchNormalization()(y44)
    y46 = LeakyReLU()(y45)
    y47 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y46)
    y48 = BatchNormalization()(y47)
    y49 = LeakyReLU()(y48)
    y50 = Add()([y46, y49])
    y51 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y41)
    y52 = BatchNormalization()(y51)
    y53 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y52)
    y54 = BatchNormalization()(y53)
    y55 = LeakyReLU()(y54)
    y56 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y55)
    y57 = BatchNormalization()(y56)
    y58 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y57)
    y59 = BatchNormalization()(y58)
    y60 = LeakyReLU()(y59)
    y61 = Add()([y60, y55])
    y62 = Concatenate()([y50, y61])
    y63 = GlobalAveragePooling2D()(y62)
    y64 = Flatten()(y63)
    y65 = Dense(1024, use_bias=False)(y64)
    y66 = BatchNormalization()(y65)
    y67 = LeakyReLU()(y66)
    y68 = Dense(10, activation="softmax")(y67)
    y69 =  (y68)
    return Model(inputs=y0, outputs=y69)
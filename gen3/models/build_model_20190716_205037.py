def build_model_20190716_205037(num_channels):
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
    y10 = Concatenate()([y3, y9])
    y11 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y10)
    y12 = BatchNormalization()(y11)
    y13 = LeakyReLU()(y12)
    y14 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y13)
    y15 = BatchNormalization()(y14)
    y16 = LeakyReLU()(y15)
    y17 = Conv2D(2*num_channels, (1,1), padding='same', use_bias=False)(y16)
    y18 = BatchNormalization()(y17)
    y19 = LeakyReLU()(y18)
    y20 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y19)
    y21 = BatchNormalization()(y20)
    y22 = LeakyReLU()(y21)
    y23 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y22)
    y24 = BatchNormalization()(y23)
    y25 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y24)
    y26 = BatchNormalization()(y25)
    y27 = LeakyReLU()(y26)
    y28 = Add()([y27, y16])
    y29 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y10)
    y30 = BatchNormalization()(y29)
    y31 = LeakyReLU()(y30)
    y32 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y31)
    y33 = BatchNormalization()(y32)
    y34 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y33)
    y35 = BatchNormalization()(y34)
    y36 = LeakyReLU()(y35)
    y37 = Add()([y36, y31])
    y38 = Concatenate()([y28, y37])
    y39 = MaxPooling2D(2,2,padding='same')(y38)
    y40 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y39)
    y41 = BatchNormalization()(y40)
    y42 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y41)
    y43 = BatchNormalization()(y42)
    y44 = LeakyReLU()(y43)
    y45 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y44)
    y46 = BatchNormalization()(y45)
    y47 = LeakyReLU()(y46)
    y48 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y44)
    y49 = BatchNormalization()(y48)
    y50 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y49)
    y51 = BatchNormalization()(y50)
    y52 = LeakyReLU()(y51)
    y53 = Concatenate()([y47, y52])
    y54 = Conv2D(16*num_channels, (1,1), padding='same', use_bias=False)(y53)
    y55 = BatchNormalization()(y54)
    y56 = LeakyReLU()(y55)
    y57 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y56)
    y58 = BatchNormalization()(y57)
    y59 = LeakyReLU()(y58)
    y60 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y59)
    y61 = BatchNormalization()(y60)
    y62 = LeakyReLU()(y61)
    y63 = Add()([y59, y62])
    y64 = Concatenate()([y53, y63])
    y65 = GlobalMaxPooling2D()(y64)
    y66 = Flatten()(y65)
    y67 = Dense(1024, use_bias=False)(y66)
    y68 = BatchNormalization()(y67)
    y69 = LeakyReLU()(y68)
    y70 = Dense(10, activation="softmax")(y69)
    y71 =  (y70)
    return Model(inputs=y0, outputs=y71)
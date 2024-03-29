def build_model_20190713_170912(num_channels):
    y0 = Input(shape=(32,32,3))
    y1 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y0)
    y2 = BatchNormalization()(y1)
    y3 = LeakyReLU()(y2)
    y4 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y3)
    y5 = BatchNormalization()(y4)
    y6 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y5)
    y7 = BatchNormalization()(y6)
    y8 = LeakyReLU()(y7)
    y9 = Concatenate()([y3, y8])
    y10 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y9)
    y11 = BatchNormalization()(y10)
    y12 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y11)
    y13 = BatchNormalization()(y12)
    y14 = LeakyReLU()(y13)
    y15 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y14)
    y16 = BatchNormalization()(y15)
    y17 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y16)
    y18 = BatchNormalization()(y17)
    y19 = LeakyReLU()(y18)
    y20 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y19)
    y21 = BatchNormalization()(y20)
    y22 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y21)
    y23 = BatchNormalization()(y22)
    y24 = LeakyReLU()(y23)
    y25 = Add()([y19, y24])
    y26 = Conv2D(2*num_channels, (1,1), padding='same', use_bias=False)(y9)
    y27 = BatchNormalization()(y26)
    y28 = LeakyReLU()(y27)
    y29 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y28)
    y30 = BatchNormalization()(y29)
    y31 = LeakyReLU()(y30)
    y32 = Conv2D(2*num_channels, (1,1), padding='same', use_bias=False)(y31)
    y33 = BatchNormalization()(y32)
    y34 = LeakyReLU()(y33)
    y35 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y34)
    y36 = BatchNormalization()(y35)
    y37 = LeakyReLU()(y36)
    y38 = Add()([y31, y37])
    y39 = Concatenate()([y25, y38])
    y40 = MaxPooling2D(2,2,padding='same')(y39)
    y41 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y40)
    y42 = BatchNormalization()(y41)
    y43 = LeakyReLU()(y42)
    y44 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y43)
    y45 = BatchNormalization()(y44)
    y46 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y45)
    y47 = BatchNormalization()(y46)
    y48 = LeakyReLU()(y47)
    y49 = Concatenate()([y48, y43])
    y50 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y49)
    y51 = BatchNormalization()(y50)
    y52 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y51)
    y53 = BatchNormalization()(y52)
    y54 = LeakyReLU()(y53)
    y55 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y54)
    y56 = BatchNormalization()(y55)
    y57 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y56)
    y58 = BatchNormalization()(y57)
    y59 = LeakyReLU()(y58)
    y60 = Add()([y54, y59])
    y61 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y49)
    y62 = BatchNormalization()(y61)
    y63 = LeakyReLU()(y62)
    y64 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y63)
    y65 = BatchNormalization()(y64)
    y66 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y65)
    y67 = BatchNormalization()(y66)
    y68 = LeakyReLU()(y67)
    y69 = Concatenate()([y60, y68])
    y70 = GlobalMaxPooling2D()(y69)
    y71 = Flatten()(y70)
    y72 = Dense(10, activation="softmax")(y71)
    y73 =  (y72)
    return Model(inputs=y0, outputs=y73)
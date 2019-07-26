def build_model_20190714_122331(num_channels):
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
    y10 = Conv2D(1*num_channels, (1,1), padding='same', use_bias=False)(y9)
    y11 = BatchNormalization()(y10)
    y12 = LeakyReLU()(y11)
    y13 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y12)
    y14 = BatchNormalization()(y13)
    y15 = LeakyReLU()(y14)
    y16 = Conv2D(1*num_channels, (1,1), padding='same', use_bias=False)(y15)
    y17 = BatchNormalization()(y16)
    y18 = LeakyReLU()(y17)
    y19 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y18)
    y20 = BatchNormalization()(y19)
    y21 = LeakyReLU()(y20)
    y22 = Add()([y15, y21])
    y23 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y9)
    y24 = BatchNormalization()(y23)
    y25 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y24)
    y26 = BatchNormalization()(y25)
    y27 = LeakyReLU()(y26)
    y28 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y27)
    y29 = BatchNormalization()(y28)
    y30 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y29)
    y31 = BatchNormalization()(y30)
    y32 = LeakyReLU()(y31)
    y33 = Conv2D(1*num_channels, (1,1), padding='same', use_bias=False)(y32)
    y34 = BatchNormalization()(y33)
    y35 = LeakyReLU()(y34)
    y36 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y35)
    y37 = BatchNormalization()(y36)
    y38 = LeakyReLU()(y37)
    y39 = Conv2D(1*num_channels, (1,1), padding='same', use_bias=False)(y38)
    y40 = BatchNormalization()(y39)
    y41 = LeakyReLU()(y40)
    y42 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y41)
    y43 = BatchNormalization()(y42)
    y44 = LeakyReLU()(y43)
    y45 = Add()([y44, y38])
    y46 = Add()([y45, y32])
    y47 = Concatenate()([y22, y46])
    y48 = MaxPooling2D(2,2,padding='same')(y47)
    y49 = Conv2D(4*num_channels, (1,1), padding='same', use_bias=False)(y48)
    y50 = BatchNormalization()(y49)
    y51 = LeakyReLU()(y50)
    y52 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y51)
    y53 = BatchNormalization()(y52)
    y54 = LeakyReLU()(y53)
    y55 = Conv2D(4*num_channels, (1,1), padding='same', use_bias=False)(y54)
    y56 = BatchNormalization()(y55)
    y57 = LeakyReLU()(y56)
    y58 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y57)
    y59 = BatchNormalization()(y58)
    y60 = LeakyReLU()(y59)
    y61 = Conv2D(4*num_channels, (1,1), padding='same', use_bias=False)(y60)
    y62 = BatchNormalization()(y61)
    y63 = LeakyReLU()(y62)
    y64 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y63)
    y65 = BatchNormalization()(y64)
    y66 = LeakyReLU()(y65)
    y67 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y66)
    y68 = BatchNormalization()(y67)
    y69 = LeakyReLU()(y68)
    y70 = GlobalAveragePooling2D()(y69)
    y71 = Flatten()(y70)
    y72 = Dense(10, activation="softmax")(y71)
    y73 =  (y72)
    return Model(inputs=y0, outputs=y73)
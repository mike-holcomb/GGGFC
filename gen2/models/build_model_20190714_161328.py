def build_model_20190714_161328(num_channels):
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
    y11 = LeakyReLU()(y10)
    y12 = Add()([y8, y11])
    y13 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y5)
    y14 = BatchNormalization()(y13)
    y15 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y14)
    y16 = BatchNormalization()(y15)
    y17 = LeakyReLU()(y16)
    y18 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y5)
    y19 = BatchNormalization()(y18)
    y20 = LeakyReLU()(y19)
    y21 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y20)
    y22 = BatchNormalization()(y21)
    y23 = LeakyReLU()(y22)
    y24 = Add()([y23, y20])
    y25 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y5)
    y26 = BatchNormalization()(y25)
    y27 = LeakyReLU()(y26)
    y28 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y27)
    y29 = BatchNormalization()(y28)
    y30 = LeakyReLU()(y29)
    y31 = Concatenate()([y12, y17, y24, y30])
    y32 = MaxPooling2D(2,2,padding='same')(y31)
    y33 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y32)
    y34 = BatchNormalization()(y33)
    y35 = LeakyReLU()(y34)
    y36 = Conv2D(8*num_channels, (1,1), padding='same', use_bias=False)(y35)
    y37 = BatchNormalization()(y36)
    y38 = LeakyReLU()(y37)
    y39 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y38)
    y40 = BatchNormalization()(y39)
    y41 = LeakyReLU()(y40)
    y42 = MaxPooling2D(2,2,padding='same')(y41)
    y43 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y42)
    y44 = BatchNormalization()(y43)
    y45 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y44)
    y46 = BatchNormalization()(y45)
    y47 = LeakyReLU()(y46)
    y48 = Conv2D(16*num_channels, (1,1), padding='same', use_bias=False)(y47)
    y49 = BatchNormalization()(y48)
    y50 = LeakyReLU()(y49)
    y51 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y50)
    y52 = BatchNormalization()(y51)
    y53 = LeakyReLU()(y52)
    y54 = Concatenate()([y53, y47])
    y55 = Conv2D(64*num_channels, (3,3), (2,2), padding='same', use_bias=False)(y54)
    y56 = BatchNormalization()(y55)
    y57 = LeakyReLU()(y56)
    y58 = Conv2D(64*num_channels, (1,1), padding='same', use_bias=False)(y57)
    y59 = BatchNormalization()(y58)
    y60 = LeakyReLU()(y59)
    y61 = Conv2D(64*num_channels, (3,3), padding='same', use_bias=False)(y60)
    y62 = BatchNormalization()(y61)
    y63 = LeakyReLU()(y62)
    y64 = Conv2D(64*num_channels, (3,3), padding='same', use_bias=False)(y63)
    y65 = BatchNormalization()(y64)
    y66 = Conv2D(64*num_channels, (3,3), padding='same', use_bias=False)(y65)
    y67 = BatchNormalization()(y66)
    y68 = LeakyReLU()(y67)
    y69 = Concatenate()([y68, y63])
    y70 = GlobalMaxPooling2D()(y69)
    y71 = Flatten()(y70)
    y72 = Dense(1024, use_bias=False)(y71)
    y73 = BatchNormalization()(y72)
    y74 = LeakyReLU()(y73)
    y75 = Dense(10, activation="softmax")(y74)
    y76 =  (y75)
    return Model(inputs=y0, outputs=y76)
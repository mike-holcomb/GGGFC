def build_model_20190712_092405(num_channels):
    y0 = Input(shape=(32,32,3))
    y1 = Conv2D(1*num_channels, (1,1), padding='same', use_bias=False)(y0)
    y2 = BatchNormalization()(y1)
    y3 = LeakyReLU()(y2)
    y4 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y3)
    y5 = BatchNormalization()(y4)
    y6 = LeakyReLU()(y5)
    y7 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y6)
    y8 = BatchNormalization()(y7)
    y9 = LeakyReLU()(y8)
    y10 = Concatenate()([y9, y6])
    y11 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y10)
    y12 = BatchNormalization()(y11)
    y13 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y12)
    y14 = BatchNormalization()(y13)
    y15 = LeakyReLU()(y14)
    y16 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y15)
    y17 = BatchNormalization()(y16)
    y18 = LeakyReLU()(y17)
    y19 = Add()([y15, y18])
    y20 = MaxPooling2D(2,2,padding='same')(y19)
    y21 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y20)
    y22 = BatchNormalization()(y21)
    y23 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y22)
    y24 = BatchNormalization()(y23)
    y25 = LeakyReLU()(y24)
    y26 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y25)
    y27 = BatchNormalization()(y26)
    y28 = LeakyReLU()(y27)
    y29 = Conv2D(4*num_channels, (1,1), padding='same', use_bias=False)(y28)
    y30 = BatchNormalization()(y29)
    y31 = LeakyReLU()(y30)
    y32 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y31)
    y33 = BatchNormalization()(y32)
    y34 = LeakyReLU()(y33)
    y35 = Add()([y28, y34])
    y36 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y25)
    y37 = BatchNormalization()(y36)
    y38 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y37)
    y39 = BatchNormalization()(y38)
    y40 = LeakyReLU()(y39)
    y41 = Conv2D(4*num_channels, (1,1), padding='same', use_bias=False)(y25)
    y42 = BatchNormalization()(y41)
    y43 = LeakyReLU()(y42)
    y44 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y43)
    y45 = BatchNormalization()(y44)
    y46 = LeakyReLU()(y45)
    y47 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y25)
    y48 = BatchNormalization()(y47)
    y49 = LeakyReLU()(y48)
    y50 = Concatenate()([y35, y40, y46, y49])
    y51 = Conv2D(32*num_channels, (3,3), (2,2), padding='same', use_bias=False)(y50)
    y52 = BatchNormalization()(y51)
    y53 = LeakyReLU()(y52)
    y54 = Conv2D(32*num_channels, (3,3), padding='same', use_bias=False)(y53)
    y55 = BatchNormalization()(y54)
    y56 = LeakyReLU()(y55)
    y57 = Conv2D(32*num_channels, (1,1), padding='same', use_bias=False)(y56)
    y58 = BatchNormalization()(y57)
    y59 = LeakyReLU()(y58)
    y60 = Conv2D(32*num_channels, (3,3), padding='same', use_bias=False)(y59)
    y61 = BatchNormalization()(y60)
    y62 = LeakyReLU()(y61)
    y63 = Concatenate()([y62, y56])
    y64 = MaxPooling2D(2,2,padding='same')(y63)
    y65 = Conv2D(128*num_channels, (3,3), padding='same', use_bias=False)(y64)
    y66 = BatchNormalization()(y65)
    y67 = Conv2D(128*num_channels, (3,3), padding='same', use_bias=False)(y66)
    y68 = BatchNormalization()(y67)
    y69 = LeakyReLU()(y68)
    y70 = Conv2D(128*num_channels, (3,3), padding='same', use_bias=False)(y69)
    y71 = BatchNormalization()(y70)
    y72 = Conv2D(128*num_channels, (3,3), padding='same', use_bias=False)(y71)
    y73 = BatchNormalization()(y72)
    y74 = LeakyReLU()(y73)
    y75 = Conv2D(128*num_channels, (3,3), padding='same', use_bias=False)(y69)
    y76 = BatchNormalization()(y75)
    y77 = Conv2D(128*num_channels, (3,3), padding='same', use_bias=False)(y76)
    y78 = BatchNormalization()(y77)
    y79 = LeakyReLU()(y78)
    y80 = Concatenate()([y74, y79])
    y81 = GlobalAveragePooling2D()(y80)
    y82 = Flatten()(y81)
    y83 = Dense(1024, use_bias=False)(y82)
    y84 = BatchNormalization()(y83)
    y85 = LeakyReLU()(y84)
    y86 = Dense(10, activation="softmax")(y85)
    y87 =  (y86)
    return Model(inputs=y0, outputs=y87)
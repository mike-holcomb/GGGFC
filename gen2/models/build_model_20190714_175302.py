def build_model_20190714_175302(num_channels):
    y0 = Input(shape=(32,32,3))
    y1 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y0)
    y2 = BatchNormalization()(y1)
    y3 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y2)
    y4 = BatchNormalization()(y3)
    y5 = LeakyReLU()(y4)
    y6 = Conv2D(1*num_channels, (1,1), padding='same', use_bias=False)(y5)
    y7 = BatchNormalization()(y6)
    y8 = LeakyReLU()(y7)
    y9 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y8)
    y10 = BatchNormalization()(y9)
    y11 = LeakyReLU()(y10)
    y12 = Concatenate()([y11, y5])
    y13 = Conv2D(2*num_channels, (1,1), padding='same', use_bias=False)(y12)
    y14 = BatchNormalization()(y13)
    y15 = LeakyReLU()(y14)
    y16 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y15)
    y17 = BatchNormalization()(y16)
    y18 = LeakyReLU()(y17)
    y19 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y18)
    y20 = BatchNormalization()(y19)
    y21 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y20)
    y22 = BatchNormalization()(y21)
    y23 = LeakyReLU()(y22)
    y24 = MaxPooling2D(2,2,padding='same')(y23)
    y25 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y24)
    y26 = BatchNormalization()(y25)
    y27 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y26)
    y28 = BatchNormalization()(y27)
    y29 = LeakyReLU()(y28)
    y30 = Conv2D(4*num_channels, (1,1), padding='same', use_bias=False)(y29)
    y31 = BatchNormalization()(y30)
    y32 = LeakyReLU()(y31)
    y33 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y32)
    y34 = BatchNormalization()(y33)
    y35 = LeakyReLU()(y34)
    y36 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y29)
    y37 = BatchNormalization()(y36)
    y38 = LeakyReLU()(y37)
    y39 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y29)
    y40 = BatchNormalization()(y39)
    y41 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y40)
    y42 = BatchNormalization()(y41)
    y43 = LeakyReLU()(y42)
    y44 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y43)
    y45 = BatchNormalization()(y44)
    y46 = LeakyReLU()(y45)
    y47 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y29)
    y48 = BatchNormalization()(y47)
    y49 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y48)
    y50 = BatchNormalization()(y49)
    y51 = LeakyReLU()(y50)
    y52 = Concatenate()([y35, y38, y46, y51])
    y53 = Conv2D(32*num_channels, (3,3), (2,2), padding='same', use_bias=False)(y52)
    y54 = BatchNormalization()(y53)
    y55 = LeakyReLU()(y54)
    y56 = Conv2D(32*num_channels, (3,3), padding='same', use_bias=False)(y55)
    y57 = BatchNormalization()(y56)
    y58 = Conv2D(32*num_channels, (3,3), padding='same', use_bias=False)(y57)
    y59 = BatchNormalization()(y58)
    y60 = LeakyReLU()(y59)
    y61 = Conv2D(32*num_channels, (3,3), padding='same', use_bias=False)(y60)
    y62 = BatchNormalization()(y61)
    y63 = Conv2D(32*num_channels, (3,3), padding='same', use_bias=False)(y62)
    y64 = BatchNormalization()(y63)
    y65 = LeakyReLU()(y64)
    y66 = Concatenate()([y60, y65])
    y67 = Conv2D(128*num_channels, (3,3), (2,2), padding='same', use_bias=False)(y66)
    y68 = BatchNormalization()(y67)
    y69 = LeakyReLU()(y68)
    y70 = Conv2D(128*num_channels, (3,3), padding='same', use_bias=False)(y69)
    y71 = BatchNormalization()(y70)
    y72 = Conv2D(128*num_channels, (3,3), padding='same', use_bias=False)(y71)
    y73 = BatchNormalization()(y72)
    y74 = LeakyReLU()(y73)
    y75 = Conv2D(128*num_channels, (3,3), padding='same', use_bias=False)(y74)
    y76 = BatchNormalization()(y75)
    y77 = Conv2D(128*num_channels, (3,3), padding='same', use_bias=False)(y76)
    y78 = BatchNormalization()(y77)
    y79 = LeakyReLU()(y78)
    y80 = Concatenate()([y79, y74])
    y81 = GlobalAveragePooling2D()(y80)
    y82 = Flatten()(y81)
    y83 = Dense(10, activation="softmax")(y82)
    y84 =  (y83)
    return Model(inputs=y0, outputs=y84)
def build_model_20190716_125450(num_channels):
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
    y10 = Conv2D(1*num_channels, (1,1), padding='same', use_bias=False)(y3)
    y11 = BatchNormalization()(y10)
    y12 = LeakyReLU()(y11)
    y13 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y12)
    y14 = BatchNormalization()(y13)
    y15 = LeakyReLU()(y14)
    y16 = Conv2D(1*num_channels, (1,1), padding='same', use_bias=False)(y3)
    y17 = BatchNormalization()(y16)
    y18 = LeakyReLU()(y17)
    y19 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y18)
    y20 = BatchNormalization()(y19)
    y21 = LeakyReLU()(y20)
    y22 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y3)
    y23 = BatchNormalization()(y22)
    y24 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y23)
    y25 = BatchNormalization()(y24)
    y26 = LeakyReLU()(y25)
    y27 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y26)
    y28 = BatchNormalization()(y27)
    y29 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y28)
    y30 = BatchNormalization()(y29)
    y31 = LeakyReLU()(y30)
    y32 = Add()([y26, y31])
    y33 = Concatenate()([y9, y15, y21, y32])
    y34 = Conv2D(4*num_channels, (1,1), padding='same', use_bias=False)(y33)
    y35 = BatchNormalization()(y34)
    y36 = LeakyReLU()(y35)
    y37 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y36)
    y38 = BatchNormalization()(y37)
    y39 = LeakyReLU()(y38)
    y40 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y39)
    y41 = BatchNormalization()(y40)
    y42 = LeakyReLU()(y41)
    y43 = Conv2D(4*num_channels, (1,1), padding='same', use_bias=False)(y42)
    y44 = BatchNormalization()(y43)
    y45 = LeakyReLU()(y44)
    y46 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y45)
    y47 = BatchNormalization()(y46)
    y48 = LeakyReLU()(y47)
    y49 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y48)
    y50 = BatchNormalization()(y49)
    y51 = LeakyReLU()(y50)
    y52 = Add()([y51, y48])
    y53 = Add()([y42, y52])
    y54 = Conv2D(8*num_channels, (3,3), (2,2), padding='same', use_bias=False)(y53)
    y55 = BatchNormalization()(y54)
    y56 = LeakyReLU()(y55)
    y57 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y56)
    y58 = BatchNormalization()(y57)
    y59 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y58)
    y60 = BatchNormalization()(y59)
    y61 = LeakyReLU()(y60)
    y62 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y61)
    y63 = BatchNormalization()(y62)
    y64 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y63)
    y65 = BatchNormalization()(y64)
    y66 = LeakyReLU()(y65)
    y67 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y61)
    y68 = BatchNormalization()(y67)
    y69 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y68)
    y70 = BatchNormalization()(y69)
    y71 = LeakyReLU()(y70)
    y72 = Concatenate()([y66, y71])
    y73 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y72)
    y74 = BatchNormalization()(y73)
    y75 = LeakyReLU()(y74)
    y76 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y75)
    y77 = BatchNormalization()(y76)
    y78 = LeakyReLU()(y77)
    y79 = GlobalAveragePooling2D()(y78)
    y80 = Flatten()(y79)
    y81 = Dense(1024, use_bias=False)(y80)
    y82 = BatchNormalization()(y81)
    y83 = LeakyReLU()(y82)
    y84 = Dense(10, activation="softmax")(y83)
    y85 =  (y84)
    return Model(inputs=y0, outputs=y85)
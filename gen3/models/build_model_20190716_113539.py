def build_model_20190716_113539(num_channels):
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
    y13 = LeakyReLU()(y12)
    y14 = Conv2D(2*num_channels, (1,1), padding='same', use_bias=False)(y13)
    y15 = BatchNormalization()(y14)
    y16 = LeakyReLU()(y15)
    y17 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y16)
    y18 = BatchNormalization()(y17)
    y19 = LeakyReLU()(y18)
    y20 = Conv2D(2*num_channels, (1,1), padding='same', use_bias=False)(y19)
    y21 = BatchNormalization()(y20)
    y22 = LeakyReLU()(y21)
    y23 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y22)
    y24 = BatchNormalization()(y23)
    y25 = LeakyReLU()(y24)
    y26 = Conv2D(2*num_channels, (1,1), padding='same', use_bias=False)(y10)
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
    y38 = Concatenate()([y25, y37])
    y39 = Conv2D(8*num_channels, (3,3), (2,2), padding='same', use_bias=False)(y38)
    y40 = BatchNormalization()(y39)
    y41 = LeakyReLU()(y40)
    y42 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y41)
    y43 = BatchNormalization()(y42)
    y44 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y43)
    y45 = BatchNormalization()(y44)
    y46 = LeakyReLU()(y45)
    y47 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y46)
    y48 = BatchNormalization()(y47)
    y49 = LeakyReLU()(y48)
    y50 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y49)
    y51 = BatchNormalization()(y50)
    y52 = LeakyReLU()(y51)
    y53 = Conv2D(8*num_channels, (1,1), padding='same', use_bias=False)(y52)
    y54 = BatchNormalization()(y53)
    y55 = LeakyReLU()(y54)
    y56 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y55)
    y57 = BatchNormalization()(y56)
    y58 = LeakyReLU()(y57)
    y59 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y49)
    y60 = BatchNormalization()(y59)
    y61 = LeakyReLU()(y60)
    y62 = Conv2D(8*num_channels, (1,1), padding='same', use_bias=False)(y61)
    y63 = BatchNormalization()(y62)
    y64 = LeakyReLU()(y63)
    y65 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y64)
    y66 = BatchNormalization()(y65)
    y67 = LeakyReLU()(y66)
    y68 = Concatenate()([y58, y67])
    y69 = GlobalMaxPooling2D()(y68)
    y70 = Flatten()(y69)
    y71 = Dense(1024, use_bias=False)(y70)
    y72 = BatchNormalization()(y71)
    y73 = LeakyReLU()(y72)
    y74 = Dense(10, activation="softmax")(y73)
    y75 =  (y74)
    return Model(inputs=y0, outputs=y75)
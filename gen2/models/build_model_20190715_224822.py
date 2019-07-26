def build_model_20190715_224822(num_channels):
    y0 = Input(shape=(32,32,3))
    y1 = Conv2D(1*num_channels, (1,1), padding='same', use_bias=False)(y0)
    y2 = BatchNormalization()(y1)
    y3 = LeakyReLU()(y2)
    y4 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y3)
    y5 = BatchNormalization()(y4)
    y6 = LeakyReLU()(y5)
    y7 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y6)
    y8 = BatchNormalization()(y7)
    y9 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y8)
    y10 = BatchNormalization()(y9)
    y11 = LeakyReLU()(y10)
    y12 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y11)
    y13 = BatchNormalization()(y12)
    y14 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y13)
    y15 = BatchNormalization()(y14)
    y16 = LeakyReLU()(y15)
    y17 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y16)
    y18 = BatchNormalization()(y17)
    y19 = LeakyReLU()(y18)
    y20 = Add()([y16, y19])
    y21 = MaxPooling2D(2,2,padding='same')(y20)
    y22 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y21)
    y23 = BatchNormalization()(y22)
    y24 = LeakyReLU()(y23)
    y25 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y24)
    y26 = BatchNormalization()(y25)
    y27 = LeakyReLU()(y26)
    y28 = Conv2D(2*num_channels, (1,1), padding='same', use_bias=False)(y27)
    y29 = BatchNormalization()(y28)
    y30 = LeakyReLU()(y29)
    y31 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y30)
    y32 = BatchNormalization()(y31)
    y33 = LeakyReLU()(y32)
    y34 = MaxPooling2D(2,2,padding='same')(y33)
    y35 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y34)
    y36 = BatchNormalization()(y35)
    y37 = LeakyReLU()(y36)
    y38 = Conv2D(4*num_channels, (1,1), padding='same', use_bias=False)(y37)
    y39 = BatchNormalization()(y38)
    y40 = LeakyReLU()(y39)
    y41 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y40)
    y42 = BatchNormalization()(y41)
    y43 = LeakyReLU()(y42)
    y44 = MaxPooling2D(2,2,padding='same')(y43)
    y45 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y44)
    y46 = BatchNormalization()(y45)
    y47 = LeakyReLU()(y46)
    y48 = Conv2D(8*num_channels, (1,1), padding='same', use_bias=False)(y47)
    y49 = BatchNormalization()(y48)
    y50 = LeakyReLU()(y49)
    y51 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y50)
    y52 = BatchNormalization()(y51)
    y53 = LeakyReLU()(y52)
    y54 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y47)
    y55 = BatchNormalization()(y54)
    y56 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y55)
    y57 = BatchNormalization()(y56)
    y58 = LeakyReLU()(y57)
    y59 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y47)
    y60 = BatchNormalization()(y59)
    y61 = LeakyReLU()(y60)
    y62 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y47)
    y63 = BatchNormalization()(y62)
    y64 = LeakyReLU()(y63)
    y65 = Concatenate()([y53, y58, y61, y64])
    y66 = GlobalMaxPooling2D()(y65)
    y67 = Flatten()(y66)
    y68 = Dense(1024, use_bias=False)(y67)
    y69 = BatchNormalization()(y68)
    y70 = LeakyReLU()(y69)
    y71 = Dense(10, activation="softmax")(y70)
    y72 =  (y71)
    return Model(inputs=y0, outputs=y72)
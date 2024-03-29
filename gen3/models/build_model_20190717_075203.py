def build_model_20190717_075203(num_channels):
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
    y16 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y15)
    y17 = BatchNormalization()(y16)
    y18 = LeakyReLU()(y17)
    y19 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y18)
    y20 = BatchNormalization()(y19)
    y21 = LeakyReLU()(y20)
    y22 = Add()([y18, y21])
    y23 = Conv2D(4*num_channels, (3,3), (2,2), padding='same', use_bias=False)(y22)
    y24 = BatchNormalization()(y23)
    y25 = LeakyReLU()(y24)
    y26 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y25)
    y27 = BatchNormalization()(y26)
    y28 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y27)
    y29 = BatchNormalization()(y28)
    y30 = LeakyReLU()(y29)
    y31 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y30)
    y32 = BatchNormalization()(y31)
    y33 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y32)
    y34 = BatchNormalization()(y33)
    y35 = LeakyReLU()(y34)
    y36 = Conv2D(4*num_channels, (1,1), padding='same', use_bias=False)(y30)
    y37 = BatchNormalization()(y36)
    y38 = LeakyReLU()(y37)
    y39 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y38)
    y40 = BatchNormalization()(y39)
    y41 = LeakyReLU()(y40)
    y42 = Concatenate()([y35, y41])
    y43 = Conv2D(8*num_channels, (1,1), padding='same', use_bias=False)(y42)
    y44 = BatchNormalization()(y43)
    y45 = LeakyReLU()(y44)
    y46 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y45)
    y47 = BatchNormalization()(y46)
    y48 = LeakyReLU()(y47)
    y49 = Conv2D(8*num_channels, (1,1), padding='same', use_bias=False)(y48)
    y50 = BatchNormalization()(y49)
    y51 = LeakyReLU()(y50)
    y52 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y51)
    y53 = BatchNormalization()(y52)
    y54 = LeakyReLU()(y53)
    y55 = Add()([y54, y48])
    y56 = Concatenate()([y42, y55])
    y57 = MaxPooling2D(2,2,padding='same')(y56)
    y58 = Conv2D(32*num_channels, (3,3), padding='same', use_bias=False)(y57)
    y59 = BatchNormalization()(y58)
    y60 = LeakyReLU()(y59)
    y61 = Conv2D(32*num_channels, (1,1), padding='same', use_bias=False)(y60)
    y62 = BatchNormalization()(y61)
    y63 = LeakyReLU()(y62)
    y64 = Conv2D(32*num_channels, (3,3), padding='same', use_bias=False)(y63)
    y65 = BatchNormalization()(y64)
    y66 = LeakyReLU()(y65)
    y67 = Conv2D(32*num_channels, (3,3), padding='same', use_bias=False)(y66)
    y68 = BatchNormalization()(y67)
    y69 = Conv2D(32*num_channels, (3,3), padding='same', use_bias=False)(y68)
    y70 = BatchNormalization()(y69)
    y71 = LeakyReLU()(y70)
    y72 = Add()([y66, y71])
    y73 = GlobalAveragePooling2D()(y72)
    y74 = Flatten()(y73)
    y75 = Dense(10, activation="softmax")(y74)
    y76 =  (y75)
    return Model(inputs=y0, outputs=y76)
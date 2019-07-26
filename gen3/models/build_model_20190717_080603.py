def build_model_20190717_080603(num_channels):
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
    y12 = LeakyReLU()(y11)
    y13 = Conv2D(2*num_channels, (1,1), padding='same', use_bias=False)(y12)
    y14 = BatchNormalization()(y13)
    y15 = LeakyReLU()(y14)
    y16 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y15)
    y17 = BatchNormalization()(y16)
    y18 = LeakyReLU()(y17)
    y19 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y9)
    y20 = BatchNormalization()(y19)
    y21 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y20)
    y22 = BatchNormalization()(y21)
    y23 = LeakyReLU()(y22)
    y24 = Conv2D(2*num_channels, (1,1), padding='same', use_bias=False)(y23)
    y25 = BatchNormalization()(y24)
    y26 = LeakyReLU()(y25)
    y27 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y26)
    y28 = BatchNormalization()(y27)
    y29 = LeakyReLU()(y28)
    y30 = Add()([y23, y29])
    y31 = Concatenate()([y18, y30])
    y32 = Conv2D(8*num_channels, (3,3), (2,2), padding='same', use_bias=False)(y31)
    y33 = BatchNormalization()(y32)
    y34 = LeakyReLU()(y33)
    y35 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y34)
    y36 = BatchNormalization()(y35)
    y37 = LeakyReLU()(y36)
    y38 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y37)
    y39 = BatchNormalization()(y38)
    y40 = LeakyReLU()(y39)
    y41 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y40)
    y42 = BatchNormalization()(y41)
    y43 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y42)
    y44 = BatchNormalization()(y43)
    y45 = LeakyReLU()(y44)
    y46 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y37)
    y47 = BatchNormalization()(y46)
    y48 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y47)
    y49 = BatchNormalization()(y48)
    y50 = LeakyReLU()(y49)
    y51 = Conv2D(8*num_channels, (1,1), padding='same', use_bias=False)(y50)
    y52 = BatchNormalization()(y51)
    y53 = LeakyReLU()(y52)
    y54 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y53)
    y55 = BatchNormalization()(y54)
    y56 = LeakyReLU()(y55)
    y57 = Add()([y50, y56])
    y58 = Concatenate()([y45, y57])
    y59 = MaxPooling2D(2,2,padding='same')(y58)
    y60 = Conv2D(32*num_channels, (1,1), padding='same', use_bias=False)(y59)
    y61 = BatchNormalization()(y60)
    y62 = LeakyReLU()(y61)
    y63 = Conv2D(32*num_channels, (3,3), padding='same', use_bias=False)(y62)
    y64 = BatchNormalization()(y63)
    y65 = LeakyReLU()(y64)
    y66 = Conv2D(32*num_channels, (3,3), padding='same', use_bias=False)(y65)
    y67 = BatchNormalization()(y66)
    y68 = Conv2D(32*num_channels, (3,3), padding='same', use_bias=False)(y67)
    y69 = BatchNormalization()(y68)
    y70 = LeakyReLU()(y69)
    y71 = Concatenate()([y70, y65])
    y72 = GlobalMaxPooling2D()(y71)
    y73 = Flatten()(y72)
    y74 = Dense(1024, use_bias=False)(y73)
    y75 = BatchNormalization()(y74)
    y76 = LeakyReLU()(y75)
    y77 = Dense(10, activation="softmax")(y76)
    y78 =  (y77)
    return Model(inputs=y0, outputs=y78)
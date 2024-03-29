def build_model_20190717_013032(num_channels):
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
    y10 = Concatenate()([y6, y9])
    y11 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y10)
    y12 = BatchNormalization()(y11)
    y13 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y12)
    y14 = BatchNormalization()(y13)
    y15 = LeakyReLU()(y14)
    y16 = Conv2D(2*num_channels, (1,1), padding='same', use_bias=False)(y15)
    y17 = BatchNormalization()(y16)
    y18 = LeakyReLU()(y17)
    y19 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y18)
    y20 = BatchNormalization()(y19)
    y21 = LeakyReLU()(y20)
    y22 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y10)
    y23 = BatchNormalization()(y22)
    y24 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y23)
    y25 = BatchNormalization()(y24)
    y26 = LeakyReLU()(y25)
    y27 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y26)
    y28 = BatchNormalization()(y27)
    y29 = LeakyReLU()(y28)
    y30 = Concatenate()([y21, y29])
    y31 = Conv2D(8*num_channels, (3,3), (2,2), padding='same', use_bias=False)(y30)
    y32 = BatchNormalization()(y31)
    y33 = LeakyReLU()(y32)
    y34 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y33)
    y35 = BatchNormalization()(y34)
    y36 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y35)
    y37 = BatchNormalization()(y36)
    y38 = LeakyReLU()(y37)
    y39 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y38)
    y40 = BatchNormalization()(y39)
    y41 = LeakyReLU()(y40)
    y42 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y41)
    y43 = BatchNormalization()(y42)
    y44 = LeakyReLU()(y43)
    y45 = Add()([y41, y44])
    y46 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y38)
    y47 = BatchNormalization()(y46)
    y48 = LeakyReLU()(y47)
    y49 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y48)
    y50 = BatchNormalization()(y49)
    y51 = LeakyReLU()(y50)
    y52 = Add()([y48, y51])
    y53 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y38)
    y54 = BatchNormalization()(y53)
    y55 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y54)
    y56 = BatchNormalization()(y55)
    y57 = LeakyReLU()(y56)
    y58 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y57)
    y59 = BatchNormalization()(y58)
    y60 = LeakyReLU()(y59)
    y61 = Add()([y57, y60])
    y62 = Conv2D(8*num_channels, (1,1), padding='same', use_bias=False)(y38)
    y63 = BatchNormalization()(y62)
    y64 = LeakyReLU()(y63)
    y65 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y64)
    y66 = BatchNormalization()(y65)
    y67 = LeakyReLU()(y66)
    y68 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y67)
    y69 = BatchNormalization()(y68)
    y70 = LeakyReLU()(y69)
    y71 = Add()([y70, y67])
    y72 = Concatenate()([y45, y52, y61, y71])
    y73 = GlobalMaxPooling2D()(y72)
    y74 = Flatten()(y73)
    y75 = Dense(1024, use_bias=False)(y74)
    y76 = BatchNormalization()(y75)
    y77 = LeakyReLU()(y76)
    y78 = Dense(10, activation="softmax")(y77)
    y79 =  (y78)
    return Model(inputs=y0, outputs=y79)
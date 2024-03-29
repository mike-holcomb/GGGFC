def build_model_20190712_092856(num_channels):
    y0 = Input(shape=(32,32,3))
    y1 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y0)
    y2 = BatchNormalization()(y1)
    y3 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y2)
    y4 = BatchNormalization()(y3)
    y5 = LeakyReLU()(y4)
    y6 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y5)
    y7 = BatchNormalization()(y6)
    y8 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y7)
    y9 = BatchNormalization()(y8)
    y10 = LeakyReLU()(y9)
    y11 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y10)
    y12 = BatchNormalization()(y11)
    y13 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y12)
    y14 = BatchNormalization()(y13)
    y15 = LeakyReLU()(y14)
    y16 = Conv2D(1*num_channels, (1,1), padding='same', use_bias=False)(y15)
    y17 = BatchNormalization()(y16)
    y18 = LeakyReLU()(y17)
    y19 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y18)
    y20 = BatchNormalization()(y19)
    y21 = LeakyReLU()(y20)
    y22 = Add()([y21, y15])
    y23 = Conv2D(2*num_channels, (3,3), (2,2), padding='same', use_bias=False)(y22)
    y24 = BatchNormalization()(y23)
    y25 = LeakyReLU()(y24)
    y26 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y25)
    y27 = BatchNormalization()(y26)
    y28 = LeakyReLU()(y27)
    y29 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y28)
    y30 = BatchNormalization()(y29)
    y31 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y30)
    y32 = BatchNormalization()(y31)
    y33 = LeakyReLU()(y32)
    y34 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y33)
    y35 = BatchNormalization()(y34)
    y36 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y35)
    y37 = BatchNormalization()(y36)
    y38 = LeakyReLU()(y37)
    y39 = Conv2D(2*num_channels, (1,1), padding='same', use_bias=False)(y28)
    y40 = BatchNormalization()(y39)
    y41 = LeakyReLU()(y40)
    y42 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y41)
    y43 = BatchNormalization()(y42)
    y44 = LeakyReLU()(y43)
    y45 = Concatenate()([y38, y44])
    y46 = MaxPooling2D(2,2,padding='same')(y45)
    y47 = Conv2D(8*num_channels, (1,1), padding='same', use_bias=False)(y46)
    y48 = BatchNormalization()(y47)
    y49 = LeakyReLU()(y48)
    y50 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y49)
    y51 = BatchNormalization()(y50)
    y52 = LeakyReLU()(y51)
    y53 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y52)
    y54 = BatchNormalization()(y53)
    y55 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y54)
    y56 = BatchNormalization()(y55)
    y57 = LeakyReLU()(y56)
    y58 = Conv2D(16*num_channels, (3,3), (2,2), padding='same', use_bias=False)(y57)
    y59 = BatchNormalization()(y58)
    y60 = LeakyReLU()(y59)
    y61 = Conv2D(16*num_channels, (1,1), padding='same', use_bias=False)(y60)
    y62 = BatchNormalization()(y61)
    y63 = LeakyReLU()(y62)
    y64 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y63)
    y65 = BatchNormalization()(y64)
    y66 = LeakyReLU()(y65)
    y67 = Conv2D(16*num_channels, (1,1), padding='same', use_bias=False)(y66)
    y68 = BatchNormalization()(y67)
    y69 = LeakyReLU()(y68)
    y70 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y69)
    y71 = BatchNormalization()(y70)
    y72 = LeakyReLU()(y71)
    y73 = Concatenate()([y66, y72])
    y74 = GlobalAveragePooling2D()(y73)
    y75 = Flatten()(y74)
    y76 = Dense(1024, use_bias=False)(y75)
    y77 = BatchNormalization()(y76)
    y78 = LeakyReLU()(y77)
    y79 = Dense(10, activation="softmax")(y78)
    y80 =  (y79)
    return Model(inputs=y0, outputs=y80)
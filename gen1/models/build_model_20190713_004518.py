def build_model_20190713_004518(num_channels):
    y0 = Input(shape=(32,32,3))
    y1 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y0)
    y2 = BatchNormalization()(y1)
    y3 = LeakyReLU()(y2)
    y4 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y3)
    y5 = BatchNormalization()(y4)
    y6 = LeakyReLU()(y5)
    y7 = Concatenate()([y6, y3])
    y8 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y7)
    y9 = BatchNormalization()(y8)
    y10 = LeakyReLU()(y9)
    y11 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y10)
    y12 = BatchNormalization()(y11)
    y13 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y12)
    y14 = BatchNormalization()(y13)
    y15 = LeakyReLU()(y14)
    y16 = Add()([y10, y15])
    y17 = Conv2D(2*num_channels, (1,1), padding='same', use_bias=False)(y7)
    y18 = BatchNormalization()(y17)
    y19 = LeakyReLU()(y18)
    y20 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y19)
    y21 = BatchNormalization()(y20)
    y22 = LeakyReLU()(y21)
    y23 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y22)
    y24 = BatchNormalization()(y23)
    y25 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y24)
    y26 = BatchNormalization()(y25)
    y27 = LeakyReLU()(y26)
    y28 = Concatenate()([y16, y27])
    y29 = MaxPooling2D(2,2,padding='same')(y28)
    y30 = Conv2D(8*num_channels, (1,1), padding='same', use_bias=False)(y29)
    y31 = BatchNormalization()(y30)
    y32 = LeakyReLU()(y31)
    y33 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y32)
    y34 = BatchNormalization()(y33)
    y35 = LeakyReLU()(y34)
    y36 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y35)
    y37 = BatchNormalization()(y36)
    y38 = LeakyReLU()(y37)
    y39 = MaxPooling2D(2,2,padding='same')(y38)
    y40 = Conv2D(16*num_channels, (1,1), padding='same', use_bias=False)(y39)
    y41 = BatchNormalization()(y40)
    y42 = LeakyReLU()(y41)
    y43 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y42)
    y44 = BatchNormalization()(y43)
    y45 = LeakyReLU()(y44)
    y46 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y45)
    y47 = BatchNormalization()(y46)
    y48 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y47)
    y49 = BatchNormalization()(y48)
    y50 = LeakyReLU()(y49)
    y51 = Concatenate()([y45, y50])
    y52 = MaxPooling2D(2,2,padding='same')(y51)
    y53 = Conv2D(64*num_channels, (3,3), padding='same', use_bias=False)(y52)
    y54 = BatchNormalization()(y53)
    y55 = LeakyReLU()(y54)
    y56 = Conv2D(64*num_channels, (3,3), padding='same', use_bias=False)(y55)
    y57 = BatchNormalization()(y56)
    y58 = LeakyReLU()(y57)
    y59 = GlobalAveragePooling2D()(y58)
    y60 = Flatten()(y59)
    y61 = Dense(10, activation="softmax")(y60)
    y62 =  (y61)
    return Model(inputs=y0, outputs=y62)
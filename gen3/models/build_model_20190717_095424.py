def build_model_20190717_095424(num_channels):
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
    y10 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y9)
    y11 = BatchNormalization()(y10)
    y12 = LeakyReLU()(y11)
    y13 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y12)
    y14 = BatchNormalization()(y13)
    y15 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y14)
    y16 = BatchNormalization()(y15)
    y17 = LeakyReLU()(y16)
    y18 = Add()([y17, y12])
    y19 = Concatenate()([y18, y9])
    y20 = MaxPooling2D(2,2,padding='same')(y19)
    y21 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y20)
    y22 = BatchNormalization()(y21)
    y23 = LeakyReLU()(y22)
    y24 = Conv2D(4*num_channels, (1,1), padding='same', use_bias=False)(y23)
    y25 = BatchNormalization()(y24)
    y26 = LeakyReLU()(y25)
    y27 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y26)
    y28 = BatchNormalization()(y27)
    y29 = LeakyReLU()(y28)
    y30 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y29)
    y31 = BatchNormalization()(y30)
    y32 = LeakyReLU()(y31)
    y33 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y32)
    y34 = BatchNormalization()(y33)
    y35 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y34)
    y36 = BatchNormalization()(y35)
    y37 = LeakyReLU()(y36)
    y38 = Concatenate()([y37, y29])
    y39 = MaxPooling2D(2,2,padding='same')(y38)
    y40 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y39)
    y41 = BatchNormalization()(y40)
    y42 = LeakyReLU()(y41)
    y43 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y42)
    y44 = BatchNormalization()(y43)
    y45 = LeakyReLU()(y44)
    y46 = Conv2D(16*num_channels, (1,1), padding='same', use_bias=False)(y45)
    y47 = BatchNormalization()(y46)
    y48 = LeakyReLU()(y47)
    y49 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y48)
    y50 = BatchNormalization()(y49)
    y51 = LeakyReLU()(y50)
    y52 = GlobalAveragePooling2D()(y51)
    y53 = Flatten()(y52)
    y54 = Dense(1024, use_bias=False)(y53)
    y55 = BatchNormalization()(y54)
    y56 = LeakyReLU()(y55)
    y57 = Dense(10, activation="softmax")(y56)
    y58 =  (y57)
    return Model(inputs=y0, outputs=y58)
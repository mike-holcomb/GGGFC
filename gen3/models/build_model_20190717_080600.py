def build_model_20190717_080600(num_channels):
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
    y10 = Conv2D(1*num_channels, (1,1), padding='same', use_bias=False)(y9)
    y11 = BatchNormalization()(y10)
    y12 = LeakyReLU()(y11)
    y13 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y12)
    y14 = BatchNormalization()(y13)
    y15 = LeakyReLU()(y14)
    y16 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y15)
    y17 = BatchNormalization()(y16)
    y18 = LeakyReLU()(y17)
    y19 = Conv2D(1*num_channels, (1,1), padding='same', use_bias=False)(y18)
    y20 = BatchNormalization()(y19)
    y21 = LeakyReLU()(y20)
    y22 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y21)
    y23 = BatchNormalization()(y22)
    y24 = LeakyReLU()(y23)
    y25 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y24)
    y26 = BatchNormalization()(y25)
    y27 = LeakyReLU()(y26)
    y28 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y27)
    y29 = BatchNormalization()(y28)
    y30 = LeakyReLU()(y29)
    y31 = Add()([y24, y30])
    y32 = MaxPooling2D(2,2,padding='same')(y31)
    y33 = Conv2D(2*num_channels, (1,1), padding='same', use_bias=False)(y32)
    y34 = BatchNormalization()(y33)
    y35 = LeakyReLU()(y34)
    y36 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y35)
    y37 = BatchNormalization()(y36)
    y38 = LeakyReLU()(y37)
    y39 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y38)
    y40 = BatchNormalization()(y39)
    y41 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y40)
    y42 = BatchNormalization()(y41)
    y43 = LeakyReLU()(y42)
    y44 = Concatenate()([y38, y43])
    y45 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y44)
    y46 = BatchNormalization()(y45)
    y47 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y46)
    y48 = BatchNormalization()(y47)
    y49 = LeakyReLU()(y48)
    y50 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y49)
    y51 = BatchNormalization()(y50)
    y52 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y51)
    y53 = BatchNormalization()(y52)
    y54 = LeakyReLU()(y53)
    y55 = Add()([y49, y54])
    y56 = GlobalMaxPooling2D()(y55)
    y57 = Flatten()(y56)
    y58 = Dense(1024, use_bias=False)(y57)
    y59 = BatchNormalization()(y58)
    y60 = LeakyReLU()(y59)
    y61 = Dense(10, activation="softmax")(y60)
    y62 =  (y61)
    return Model(inputs=y0, outputs=y62)
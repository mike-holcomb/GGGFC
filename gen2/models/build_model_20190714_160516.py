def build_model_20190714_160516(num_channels):
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
    y10 = Concatenate()([y9, y3])
    y11 = Conv2D(2*num_channels, (1,1), padding='same', use_bias=False)(y10)
    y12 = BatchNormalization()(y11)
    y13 = LeakyReLU()(y12)
    y14 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y13)
    y15 = BatchNormalization()(y14)
    y16 = LeakyReLU()(y15)
    y17 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y16)
    y18 = BatchNormalization()(y17)
    y19 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y18)
    y20 = BatchNormalization()(y19)
    y21 = LeakyReLU()(y20)
    y22 = Add()([y21, y16])
    y23 = MaxPooling2D(2,2,padding='same')(y22)
    y24 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y23)
    y25 = BatchNormalization()(y24)
    y26 = LeakyReLU()(y25)
    y27 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y26)
    y28 = BatchNormalization()(y27)
    y29 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y28)
    y30 = BatchNormalization()(y29)
    y31 = LeakyReLU()(y30)
    y32 = Conv2D(4*num_channels, (1,1), padding='same', use_bias=False)(y31)
    y33 = BatchNormalization()(y32)
    y34 = LeakyReLU()(y33)
    y35 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y34)
    y36 = BatchNormalization()(y35)
    y37 = LeakyReLU()(y36)
    y38 = Concatenate()([y37, y26])
    y39 = Conv2D(16*num_channels, (3,3), (2,2), padding='same', use_bias=False)(y38)
    y40 = BatchNormalization()(y39)
    y41 = LeakyReLU()(y40)
    y42 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y41)
    y43 = BatchNormalization()(y42)
    y44 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y43)
    y45 = BatchNormalization()(y44)
    y46 = LeakyReLU()(y45)
    y47 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y46)
    y48 = BatchNormalization()(y47)
    y49 = LeakyReLU()(y48)
    y50 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y46)
    y51 = BatchNormalization()(y50)
    y52 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y51)
    y53 = BatchNormalization()(y52)
    y54 = LeakyReLU()(y53)
    y55 = Conv2D(16*num_channels, (1,1), padding='same', use_bias=False)(y46)
    y56 = BatchNormalization()(y55)
    y57 = LeakyReLU()(y56)
    y58 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y57)
    y59 = BatchNormalization()(y58)
    y60 = LeakyReLU()(y59)
    y61 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y46)
    y62 = BatchNormalization()(y61)
    y63 = LeakyReLU()(y62)
    y64 = Concatenate()([y49, y54, y60, y63])
    y65 = MaxPooling2D(2,2,padding='same')(y64)
    y66 = Conv2D(128*num_channels, (3,3), padding='same', use_bias=False)(y65)
    y67 = BatchNormalization()(y66)
    y68 = LeakyReLU()(y67)
    y69 = Conv2D(128*num_channels, (3,3), padding='same', use_bias=False)(y68)
    y70 = BatchNormalization()(y69)
    y71 = LeakyReLU()(y70)
    y72 = GlobalMaxPooling2D()(y71)
    y73 = Flatten()(y72)
    y74 = Dense(1024, use_bias=False)(y73)
    y75 = BatchNormalization()(y74)
    y76 = LeakyReLU()(y75)
    y77 = Dense(10, activation="softmax")(y76)
    y78 =  (y77)
    return Model(inputs=y0, outputs=y78)
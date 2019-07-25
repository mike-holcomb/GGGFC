def build_model_20190712_081634(num_channels):
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
    y10 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y3)
    y11 = BatchNormalization()(y10)
    y12 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y11)
    y13 = BatchNormalization()(y12)
    y14 = LeakyReLU()(y13)
    y15 = Concatenate()([y9, y14])
    y16 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y15)
    y17 = BatchNormalization()(y16)
    y18 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y17)
    y19 = BatchNormalization()(y18)
    y20 = LeakyReLU()(y19)
    y21 = Conv2D(2*num_channels, (1,1), padding='same', use_bias=False)(y20)
    y22 = BatchNormalization()(y21)
    y23 = LeakyReLU()(y22)
    y24 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y23)
    y25 = BatchNormalization()(y24)
    y26 = LeakyReLU()(y25)
    y27 = MaxPooling2D(2,2,padding='same')(y26)
    y28 = Conv2D(4*num_channels, (1,1), padding='same', use_bias=False)(y27)
    y29 = BatchNormalization()(y28)
    y30 = LeakyReLU()(y29)
    y31 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y30)
    y32 = BatchNormalization()(y31)
    y33 = LeakyReLU()(y32)
    y34 = Conv2D(4*num_channels, (1,1), padding='same', use_bias=False)(y33)
    y35 = BatchNormalization()(y34)
    y36 = LeakyReLU()(y35)
    y37 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y36)
    y38 = BatchNormalization()(y37)
    y39 = LeakyReLU()(y38)
    y40 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y33)
    y41 = BatchNormalization()(y40)
    y42 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y41)
    y43 = BatchNormalization()(y42)
    y44 = LeakyReLU()(y43)
    y45 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y44)
    y46 = BatchNormalization()(y45)
    y47 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y46)
    y48 = BatchNormalization()(y47)
    y49 = LeakyReLU()(y48)
    y50 = Add()([y49, y44])
    y51 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y33)
    y52 = BatchNormalization()(y51)
    y53 = LeakyReLU()(y52)
    y54 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y53)
    y55 = BatchNormalization()(y54)
    y56 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y55)
    y57 = BatchNormalization()(y56)
    y58 = LeakyReLU()(y57)
    y59 = Add()([y53, y58])
    y60 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y33)
    y61 = BatchNormalization()(y60)
    y62 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y61)
    y63 = BatchNormalization()(y62)
    y64 = LeakyReLU()(y63)
    y65 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y64)
    y66 = BatchNormalization()(y65)
    y67 = LeakyReLU()(y66)
    y68 = Concatenate()([y39, y50, y59, y67])
    y69 = Conv2D(32*num_channels, (3,3), (2,2), padding='same', use_bias=False)(y68)
    y70 = BatchNormalization()(y69)
    y71 = LeakyReLU()(y70)
    y72 = Conv2D(32*num_channels, (3,3), padding='same', use_bias=False)(y71)
    y73 = BatchNormalization()(y72)
    y74 = LeakyReLU()(y73)
    y75 = Conv2D(32*num_channels, (3,3), padding='same', use_bias=False)(y74)
    y76 = BatchNormalization()(y75)
    y77 = LeakyReLU()(y76)
    y78 = Concatenate()([y77, y74])
    y79 = GlobalAveragePooling2D()(y78)
    y80 = Flatten()(y79)
    y81 = Dense(10, activation="softmax")(y80)
    y82 =  (y81)
    return Model(inputs=y0, outputs=y82)
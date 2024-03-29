def build_model_20190713_170908(num_channels):
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
    y16 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y5)
    y17 = BatchNormalization()(y16)
    y18 = LeakyReLU()(y17)
    y19 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y5)
    y20 = BatchNormalization()(y19)
    y21 = LeakyReLU()(y20)
    y22 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y21)
    y23 = BatchNormalization()(y22)
    y24 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y23)
    y25 = BatchNormalization()(y24)
    y26 = LeakyReLU()(y25)
    y27 = Conv2D(1*num_channels, (1,1), padding='same', use_bias=False)(y5)
    y28 = BatchNormalization()(y27)
    y29 = LeakyReLU()(y28)
    y30 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y29)
    y31 = BatchNormalization()(y30)
    y32 = LeakyReLU()(y31)
    y33 = Conv2D(1*num_channels, (1,1), padding='same', use_bias=False)(y32)
    y34 = BatchNormalization()(y33)
    y35 = LeakyReLU()(y34)
    y36 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y35)
    y37 = BatchNormalization()(y36)
    y38 = LeakyReLU()(y37)
    y39 = Add()([y32, y38])
    y40 = Concatenate()([y15, y18, y26, y39])
    y41 = MaxPooling2D(2,2,padding='same')(y40)
    y42 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y41)
    y43 = BatchNormalization()(y42)
    y44 = LeakyReLU()(y43)
    y45 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y44)
    y46 = BatchNormalization()(y45)
    y47 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y46)
    y48 = BatchNormalization()(y47)
    y49 = LeakyReLU()(y48)
    y50 = MaxPooling2D(2,2,padding='same')(y49)
    y51 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y50)
    y52 = BatchNormalization()(y51)
    y53 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y52)
    y54 = BatchNormalization()(y53)
    y55 = LeakyReLU()(y54)
    y56 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y55)
    y57 = BatchNormalization()(y56)
    y58 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y57)
    y59 = BatchNormalization()(y58)
    y60 = LeakyReLU()(y59)
    y61 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y55)
    y62 = BatchNormalization()(y61)
    y63 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y62)
    y64 = BatchNormalization()(y63)
    y65 = LeakyReLU()(y64)
    y66 = Concatenate()([y60, y65])
    y67 = Conv2D(64*num_channels, (3,3), (2,2), padding='same', use_bias=False)(y66)
    y68 = BatchNormalization()(y67)
    y69 = LeakyReLU()(y68)
    y70 = Conv2D(64*num_channels, (3,3), padding='same', use_bias=False)(y69)
    y71 = BatchNormalization()(y70)
    y72 = Conv2D(64*num_channels, (3,3), padding='same', use_bias=False)(y71)
    y73 = BatchNormalization()(y72)
    y74 = LeakyReLU()(y73)
    y75 = Conv2D(64*num_channels, (3,3), padding='same', use_bias=False)(y74)
    y76 = BatchNormalization()(y75)
    y77 = LeakyReLU()(y76)
    y78 = Conv2D(64*num_channels, (1,1), padding='same', use_bias=False)(y74)
    y79 = BatchNormalization()(y78)
    y80 = LeakyReLU()(y79)
    y81 = Conv2D(64*num_channels, (3,3), padding='same', use_bias=False)(y80)
    y82 = BatchNormalization()(y81)
    y83 = LeakyReLU()(y82)
    y84 = Conv2D(64*num_channels, (1,1), padding='same', use_bias=False)(y74)
    y85 = BatchNormalization()(y84)
    y86 = LeakyReLU()(y85)
    y87 = Conv2D(64*num_channels, (3,3), padding='same', use_bias=False)(y86)
    y88 = BatchNormalization()(y87)
    y89 = LeakyReLU()(y88)
    y90 = Conv2D(64*num_channels, (3,3), padding='same', use_bias=False)(y74)
    y91 = BatchNormalization()(y90)
    y92 = Conv2D(64*num_channels, (3,3), padding='same', use_bias=False)(y91)
    y93 = BatchNormalization()(y92)
    y94 = LeakyReLU()(y93)
    y95 = Concatenate()([y77, y83, y89, y94])
    y96 = GlobalMaxPooling2D()(y95)
    y97 = Flatten()(y96)
    y98 = Dense(10, activation="softmax")(y97)
    y99 =  (y98)
    return Model(inputs=y0, outputs=y99)
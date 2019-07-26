def build_model_20190714_173841(num_channels):
    y0 = Input(shape=(32,32,3))
    y1 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y0)
    y2 = BatchNormalization()(y1)
    y3 = LeakyReLU()(y2)
    y4 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y3)
    y5 = BatchNormalization()(y4)
    y6 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y5)
    y7 = BatchNormalization()(y6)
    y8 = LeakyReLU()(y7)
    y9 = Conv2D(1*num_channels, (1,1), padding='same', use_bias=False)(y8)
    y10 = BatchNormalization()(y9)
    y11 = LeakyReLU()(y10)
    y12 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y11)
    y13 = BatchNormalization()(y12)
    y14 = LeakyReLU()(y13)
    y15 = Add()([y8, y14])
    y16 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y3)
    y17 = BatchNormalization()(y16)
    y18 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y17)
    y19 = BatchNormalization()(y18)
    y20 = LeakyReLU()(y19)
    y21 = Conv2D(1*num_channels, (1,1), padding='same', use_bias=False)(y20)
    y22 = BatchNormalization()(y21)
    y23 = LeakyReLU()(y22)
    y24 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y23)
    y25 = BatchNormalization()(y24)
    y26 = LeakyReLU()(y25)
    y27 = Add()([y20, y26])
    y28 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y3)
    y29 = BatchNormalization()(y28)
    y30 = LeakyReLU()(y29)
    y31 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y3)
    y32 = BatchNormalization()(y31)
    y33 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y32)
    y34 = BatchNormalization()(y33)
    y35 = LeakyReLU()(y34)
    y36 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y35)
    y37 = BatchNormalization()(y36)
    y38 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y37)
    y39 = BatchNormalization()(y38)
    y40 = LeakyReLU()(y39)
    y41 = Concatenate()([y15, y27, y30, y40])
    y42 = MaxPooling2D(2,2,padding='same')(y41)
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
    y55 = Concatenate()([y54, y48])
    y56 = Conv2D(32*num_channels, (3,3), (2,2), padding='same', use_bias=False)(y55)
    y57 = BatchNormalization()(y56)
    y58 = LeakyReLU()(y57)
    y59 = Conv2D(32*num_channels, (3,3), padding='same', use_bias=False)(y58)
    y60 = BatchNormalization()(y59)
    y61 = Conv2D(32*num_channels, (3,3), padding='same', use_bias=False)(y60)
    y62 = BatchNormalization()(y61)
    y63 = LeakyReLU()(y62)
    y64 = Conv2D(32*num_channels, (1,1), padding='same', use_bias=False)(y63)
    y65 = BatchNormalization()(y64)
    y66 = LeakyReLU()(y65)
    y67 = Conv2D(32*num_channels, (3,3), padding='same', use_bias=False)(y66)
    y68 = BatchNormalization()(y67)
    y69 = LeakyReLU()(y68)
    y70 = Conv2D(32*num_channels, (3,3), padding='same', use_bias=False)(y63)
    y71 = BatchNormalization()(y70)
    y72 = LeakyReLU()(y71)
    y73 = Conv2D(32*num_channels, (3,3), padding='same', use_bias=False)(y63)
    y74 = BatchNormalization()(y73)
    y75 = LeakyReLU()(y74)
    y76 = Conv2D(32*num_channels, (3,3), padding='same', use_bias=False)(y63)
    y77 = BatchNormalization()(y76)
    y78 = Conv2D(32*num_channels, (3,3), padding='same', use_bias=False)(y77)
    y79 = BatchNormalization()(y78)
    y80 = LeakyReLU()(y79)
    y81 = Concatenate()([y69, y72, y75, y80])
    y82 = Conv2D(256*num_channels, (3,3), (2,2), padding='same', use_bias=False)(y81)
    y83 = BatchNormalization()(y82)
    y84 = LeakyReLU()(y83)
    y85 = Conv2D(256*num_channels, (1,1), padding='same', use_bias=False)(y84)
    y86 = BatchNormalization()(y85)
    y87 = LeakyReLU()(y86)
    y88 = Conv2D(256*num_channels, (3,3), padding='same', use_bias=False)(y87)
    y89 = BatchNormalization()(y88)
    y90 = LeakyReLU()(y89)
    y91 = Conv2D(256*num_channels, (3,3), padding='same', use_bias=False)(y90)
    y92 = BatchNormalization()(y91)
    y93 = LeakyReLU()(y92)
    y94 = GlobalAveragePooling2D()(y93)
    y95 = Flatten()(y94)
    y96 = Dense(10, activation="softmax")(y95)
    y97 =  (y96)
    return Model(inputs=y0, outputs=y97)
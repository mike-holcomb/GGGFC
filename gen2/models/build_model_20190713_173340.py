def build_model_20190713_173340(num_channels):
    y0 = Input(shape=(32,32,3))
    y1 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y0)
    y2 = BatchNormalization()(y1)
    y3 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y2)
    y4 = BatchNormalization()(y3)
    y5 = LeakyReLU()(y4)
    y6 = Conv2D(1*num_channels, (1,1), padding='same', use_bias=False)(y5)
    y7 = BatchNormalization()(y6)
    y8 = LeakyReLU()(y7)
    y9 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y8)
    y10 = BatchNormalization()(y9)
    y11 = LeakyReLU()(y10)
    y12 = Conv2D(1*num_channels, (1,1), padding='same', use_bias=False)(y11)
    y13 = BatchNormalization()(y12)
    y14 = LeakyReLU()(y13)
    y15 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y14)
    y16 = BatchNormalization()(y15)
    y17 = LeakyReLU()(y16)
    y18 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y5)
    y19 = BatchNormalization()(y18)
    y20 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y19)
    y21 = BatchNormalization()(y20)
    y22 = LeakyReLU()(y21)
    y23 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y22)
    y24 = BatchNormalization()(y23)
    y25 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y24)
    y26 = BatchNormalization()(y25)
    y27 = LeakyReLU()(y26)
    y28 = Add()([y27, y22])
    y29 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y5)
    y30 = BatchNormalization()(y29)
    y31 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y30)
    y32 = BatchNormalization()(y31)
    y33 = LeakyReLU()(y32)
    y34 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y5)
    y35 = BatchNormalization()(y34)
    y36 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y35)
    y37 = BatchNormalization()(y36)
    y38 = LeakyReLU()(y37)
    y39 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y38)
    y40 = BatchNormalization()(y39)
    y41 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y40)
    y42 = BatchNormalization()(y41)
    y43 = LeakyReLU()(y42)
    y44 = Concatenate()([y17, y28, y33, y43])
    y45 = Conv2D(8*num_channels, (3,3), (2,2), padding='same', use_bias=False)(y44)
    y46 = BatchNormalization()(y45)
    y47 = LeakyReLU()(y46)
    y48 = Conv2D(8*num_channels, (1,1), padding='same', use_bias=False)(y47)
    y49 = BatchNormalization()(y48)
    y50 = LeakyReLU()(y49)
    y51 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y50)
    y52 = BatchNormalization()(y51)
    y53 = LeakyReLU()(y52)
    y54 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y53)
    y55 = BatchNormalization()(y54)
    y56 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y55)
    y57 = BatchNormalization()(y56)
    y58 = LeakyReLU()(y57)
    y59 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y53)
    y60 = BatchNormalization()(y59)
    y61 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y60)
    y62 = BatchNormalization()(y61)
    y63 = LeakyReLU()(y62)
    y64 = Conv2D(8*num_channels, (1,1), padding='same', use_bias=False)(y53)
    y65 = BatchNormalization()(y64)
    y66 = LeakyReLU()(y65)
    y67 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y66)
    y68 = BatchNormalization()(y67)
    y69 = LeakyReLU()(y68)
    y70 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y53)
    y71 = BatchNormalization()(y70)
    y72 = LeakyReLU()(y71)
    y73 = Concatenate()([y58, y63, y69, y72])
    y74 = Conv2D(64*num_channels, (3,3), (2,2), padding='same', use_bias=False)(y73)
    y75 = BatchNormalization()(y74)
    y76 = LeakyReLU()(y75)
    y77 = Conv2D(64*num_channels, (1,1), padding='same', use_bias=False)(y76)
    y78 = BatchNormalization()(y77)
    y79 = LeakyReLU()(y78)
    y80 = Conv2D(64*num_channels, (3,3), padding='same', use_bias=False)(y79)
    y81 = BatchNormalization()(y80)
    y82 = LeakyReLU()(y81)
    y83 = Conv2D(64*num_channels, (3,3), padding='same', use_bias=False)(y82)
    y84 = BatchNormalization()(y83)
    y85 = LeakyReLU()(y84)
    y86 = MaxPooling2D(2,2,padding='same')(y85)
    y87 = Conv2D(128*num_channels, (3,3), padding='same', use_bias=False)(y86)
    y88 = BatchNormalization()(y87)
    y89 = LeakyReLU()(y88)
    y90 = Conv2D(128*num_channels, (3,3), padding='same', use_bias=False)(y89)
    y91 = BatchNormalization()(y90)
    y92 = Conv2D(128*num_channels, (3,3), padding='same', use_bias=False)(y91)
    y93 = BatchNormalization()(y92)
    y94 = LeakyReLU()(y93)
    y95 = Concatenate()([y89, y94])
    y96 = GlobalMaxPooling2D()(y95)
    y97 = Flatten()(y96)
    y98 = Dense(10, activation="softmax")(y97)
    y99 =  (y98)
    return Model(inputs=y0, outputs=y99)
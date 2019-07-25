def build_model_20190714_173845(num_channels):
    y0 = Input(shape=(32,32,3))
    y1 = Conv2D(1*num_channels, (1,1), padding='same', use_bias=False)(y0)
    y2 = BatchNormalization()(y1)
    y3 = LeakyReLU()(y2)
    y4 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y3)
    y5 = BatchNormalization()(y4)
    y6 = LeakyReLU()(y5)
    y7 = Conv2D(1*num_channels, (1,1), padding='same', use_bias=False)(y6)
    y8 = BatchNormalization()(y7)
    y9 = LeakyReLU()(y8)
    y10 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y9)
    y11 = BatchNormalization()(y10)
    y12 = LeakyReLU()(y11)
    y13 = Conv2D(1*num_channels, (1,1), padding='same', use_bias=False)(y6)
    y14 = BatchNormalization()(y13)
    y15 = LeakyReLU()(y14)
    y16 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y15)
    y17 = BatchNormalization()(y16)
    y18 = LeakyReLU()(y17)
    y19 = Concatenate()([y12, y18])
    y20 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y19)
    y21 = BatchNormalization()(y20)
    y22 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y21)
    y23 = BatchNormalization()(y22)
    y24 = LeakyReLU()(y23)
    y25 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y24)
    y26 = BatchNormalization()(y25)
    y27 = LeakyReLU()(y26)
    y28 = Add()([y24, y27])
    y29 = Conv2D(2*num_channels, (1,1), padding='same', use_bias=False)(y19)
    y30 = BatchNormalization()(y29)
    y31 = LeakyReLU()(y30)
    y32 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y31)
    y33 = BatchNormalization()(y32)
    y34 = LeakyReLU()(y33)
    y35 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y34)
    y36 = BatchNormalization()(y35)
    y37 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y36)
    y38 = BatchNormalization()(y37)
    y39 = LeakyReLU()(y38)
    y40 = Add()([y39, y34])
    y41 = Concatenate()([y28, y40])
    y42 = MaxPooling2D(2,2,padding='same')(y41)
    y43 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y42)
    y44 = BatchNormalization()(y43)
    y45 = LeakyReLU()(y44)
    y46 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y45)
    y47 = BatchNormalization()(y46)
    y48 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y47)
    y49 = BatchNormalization()(y48)
    y50 = LeakyReLU()(y49)
    y51 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y45)
    y52 = BatchNormalization()(y51)
    y53 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y52)
    y54 = BatchNormalization()(y53)
    y55 = LeakyReLU()(y54)
    y56 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y45)
    y57 = BatchNormalization()(y56)
    y58 = LeakyReLU()(y57)
    y59 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y45)
    y60 = BatchNormalization()(y59)
    y61 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y60)
    y62 = BatchNormalization()(y61)
    y63 = LeakyReLU()(y62)
    y64 = Concatenate()([y50, y55, y58, y63])
    y65 = Conv2D(32*num_channels, (3,3), padding='same', use_bias=False)(y64)
    y66 = BatchNormalization()(y65)
    y67 = Conv2D(32*num_channels, (3,3), padding='same', use_bias=False)(y66)
    y68 = BatchNormalization()(y67)
    y69 = LeakyReLU()(y68)
    y70 = Conv2D(32*num_channels, (3,3), padding='same', use_bias=False)(y69)
    y71 = BatchNormalization()(y70)
    y72 = LeakyReLU()(y71)
    y73 = Add()([y72, y69])
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
    y85 = Conv2D(64*num_channels, (3,3), padding='same', use_bias=False)(y84)
    y86 = BatchNormalization()(y85)
    y87 = LeakyReLU()(y86)
    y88 = Concatenate()([y82, y87])
    y89 = GlobalMaxPooling2D()(y88)
    y90 = Flatten()(y89)
    y91 = Dense(1024, use_bias=False)(y90)
    y92 = BatchNormalization()(y91)
    y93 = LeakyReLU()(y92)
    y94 = Dense(10, activation="softmax")(y93)
    y95 =  (y94)
    return Model(inputs=y0, outputs=y95)
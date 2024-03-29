def build_model_20190714_164202(num_channels):
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
    y10 = Conv2D(1*num_channels, (1,1), padding='same', use_bias=False)(y6)
    y11 = BatchNormalization()(y10)
    y12 = LeakyReLU()(y11)
    y13 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y12)
    y14 = BatchNormalization()(y13)
    y15 = LeakyReLU()(y14)
    y16 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y6)
    y17 = BatchNormalization()(y16)
    y18 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y17)
    y19 = BatchNormalization()(y18)
    y20 = LeakyReLU()(y19)
    y21 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y6)
    y22 = BatchNormalization()(y21)
    y23 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y22)
    y24 = BatchNormalization()(y23)
    y25 = LeakyReLU()(y24)
    y26 = Concatenate()([y9, y15, y20, y25])
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
    y38 = Add()([y37, y31])
    y39 = Conv2D(8*num_channels, (3,3), (2,2), padding='same', use_bias=False)(y38)
    y40 = BatchNormalization()(y39)
    y41 = LeakyReLU()(y40)
    y42 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y41)
    y43 = BatchNormalization()(y42)
    y44 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y43)
    y45 = BatchNormalization()(y44)
    y46 = LeakyReLU()(y45)
    y47 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y46)
    y48 = BatchNormalization()(y47)
    y49 = LeakyReLU()(y48)
    y50 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y46)
    y51 = BatchNormalization()(y50)
    y52 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y51)
    y53 = BatchNormalization()(y52)
    y54 = LeakyReLU()(y53)
    y55 = Concatenate()([y49, y54])
    y56 = Conv2D(16*num_channels, (1,1), padding='same', use_bias=False)(y55)
    y57 = BatchNormalization()(y56)
    y58 = LeakyReLU()(y57)
    y59 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y58)
    y60 = BatchNormalization()(y59)
    y61 = LeakyReLU()(y60)
    y62 = Conv2D(16*num_channels, (1,1), padding='same', use_bias=False)(y61)
    y63 = BatchNormalization()(y62)
    y64 = LeakyReLU()(y63)
    y65 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y64)
    y66 = BatchNormalization()(y65)
    y67 = LeakyReLU()(y66)
    y68 = Conv2D(32*num_channels, (3,3), (2,2), padding='same', use_bias=False)(y67)
    y69 = BatchNormalization()(y68)
    y70 = LeakyReLU()(y69)
    y71 = Conv2D(32*num_channels, (1,1), padding='same', use_bias=False)(y70)
    y72 = BatchNormalization()(y71)
    y73 = LeakyReLU()(y72)
    y74 = Conv2D(32*num_channels, (3,3), padding='same', use_bias=False)(y73)
    y75 = BatchNormalization()(y74)
    y76 = LeakyReLU()(y75)
    y77 = Conv2D(32*num_channels, (3,3), padding='same', use_bias=False)(y76)
    y78 = BatchNormalization()(y77)
    y79 = LeakyReLU()(y78)
    y80 = Conv2D(32*num_channels, (3,3), padding='same', use_bias=False)(y79)
    y81 = BatchNormalization()(y80)
    y82 = LeakyReLU()(y81)
    y83 = Conv2D(32*num_channels, (3,3), padding='same', use_bias=False)(y76)
    y84 = BatchNormalization()(y83)
    y85 = Conv2D(32*num_channels, (3,3), padding='same', use_bias=False)(y84)
    y86 = BatchNormalization()(y85)
    y87 = LeakyReLU()(y86)
    y88 = Concatenate()([y82, y87])
    y89 = GlobalAveragePooling2D()(y88)
    y90 = Flatten()(y89)
    y91 = Dense(1024, use_bias=False)(y90)
    y92 = BatchNormalization()(y91)
    y93 = LeakyReLU()(y92)
    y94 = Dense(10, activation="softmax")(y93)
    y95 =  (y94)
    return Model(inputs=y0, outputs=y95)
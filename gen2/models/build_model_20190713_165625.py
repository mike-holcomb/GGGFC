def build_model_20190713_165625(num_channels):
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
    y12 = LeakyReLU()(y11)
    y13 = Conv2D(1*num_channels, (1,1), padding='same', use_bias=False)(y3)
    y14 = BatchNormalization()(y13)
    y15 = LeakyReLU()(y14)
    y16 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y15)
    y17 = BatchNormalization()(y16)
    y18 = LeakyReLU()(y17)
    y19 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y3)
    y20 = BatchNormalization()(y19)
    y21 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y20)
    y22 = BatchNormalization()(y21)
    y23 = LeakyReLU()(y22)
    y24 = Concatenate()([y9, y12, y18, y23])
    y25 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y24)
    y26 = BatchNormalization()(y25)
    y27 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y26)
    y28 = BatchNormalization()(y27)
    y29 = LeakyReLU()(y28)
    y30 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y29)
    y31 = BatchNormalization()(y30)
    y32 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y31)
    y33 = BatchNormalization()(y32)
    y34 = LeakyReLU()(y33)
    y35 = Add()([y29, y34])
    y36 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y24)
    y37 = BatchNormalization()(y36)
    y38 = LeakyReLU()(y37)
    y39 = Conv2D(4*num_channels, (1,1), padding='same', use_bias=False)(y38)
    y40 = BatchNormalization()(y39)
    y41 = LeakyReLU()(y40)
    y42 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y41)
    y43 = BatchNormalization()(y42)
    y44 = LeakyReLU()(y43)
    y45 = Add()([y38, y44])
    y46 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y24)
    y47 = BatchNormalization()(y46)
    y48 = LeakyReLU()(y47)
    y49 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y48)
    y50 = BatchNormalization()(y49)
    y51 = LeakyReLU()(y50)
    y52 = Conv2D(4*num_channels, (1,1), padding='same', use_bias=False)(y24)
    y53 = BatchNormalization()(y52)
    y54 = LeakyReLU()(y53)
    y55 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y54)
    y56 = BatchNormalization()(y55)
    y57 = LeakyReLU()(y56)
    y58 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y57)
    y59 = BatchNormalization()(y58)
    y60 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y59)
    y61 = BatchNormalization()(y60)
    y62 = LeakyReLU()(y61)
    y63 = Add()([y62, y57])
    y64 = Concatenate()([y35, y45, y51, y63])
    y65 = Conv2D(32*num_channels, (3,3), (2,2), padding='same', use_bias=False)(y64)
    y66 = BatchNormalization()(y65)
    y67 = LeakyReLU()(y66)
    y68 = Conv2D(32*num_channels, (3,3), padding='same', use_bias=False)(y67)
    y69 = BatchNormalization()(y68)
    y70 = LeakyReLU()(y69)
    y71 = Conv2D(32*num_channels, (1,1), padding='same', use_bias=False)(y70)
    y72 = BatchNormalization()(y71)
    y73 = LeakyReLU()(y72)
    y74 = Conv2D(32*num_channels, (3,3), padding='same', use_bias=False)(y73)
    y75 = BatchNormalization()(y74)
    y76 = LeakyReLU()(y75)
    y77 = Conv2D(64*num_channels, (3,3), (2,2), padding='same', use_bias=False)(y76)
    y78 = BatchNormalization()(y77)
    y79 = LeakyReLU()(y78)
    y80 = Conv2D(64*num_channels, (3,3), padding='same', use_bias=False)(y79)
    y81 = BatchNormalization()(y80)
    y82 = Conv2D(64*num_channels, (3,3), padding='same', use_bias=False)(y81)
    y83 = BatchNormalization()(y82)
    y84 = LeakyReLU()(y83)
    y85 = Conv2D(64*num_channels, (3,3), padding='same', use_bias=False)(y84)
    y86 = BatchNormalization()(y85)
    y87 = Conv2D(64*num_channels, (3,3), padding='same', use_bias=False)(y86)
    y88 = BatchNormalization()(y87)
    y89 = LeakyReLU()(y88)
    y90 = Concatenate()([y89, y84])
    y91 = GlobalMaxPooling2D()(y90)
    y92 = Flatten()(y91)
    y93 = Dense(1024, use_bias=False)(y92)
    y94 = BatchNormalization()(y93)
    y95 = LeakyReLU()(y94)
    y96 = Dense(10, activation="softmax")(y95)
    y97 =  (y96)
    return Model(inputs=y0, outputs=y97)
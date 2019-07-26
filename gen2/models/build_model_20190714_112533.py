def build_model_20190714_112533(num_channels):
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
    y10 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y6)
    y11 = BatchNormalization()(y10)
    y12 = LeakyReLU()(y11)
    y13 = Concatenate()([y9, y12])
    y14 = Conv2D(2*num_channels, (1,1), padding='same', use_bias=False)(y13)
    y15 = BatchNormalization()(y14)
    y16 = LeakyReLU()(y15)
    y17 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y16)
    y18 = BatchNormalization()(y17)
    y19 = LeakyReLU()(y18)
    y20 = Conv2D(2*num_channels, (1,1), padding='same', use_bias=False)(y19)
    y21 = BatchNormalization()(y20)
    y22 = LeakyReLU()(y21)
    y23 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y22)
    y24 = BatchNormalization()(y23)
    y25 = LeakyReLU()(y24)
    y26 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y13)
    y27 = BatchNormalization()(y26)
    y28 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y27)
    y29 = BatchNormalization()(y28)
    y30 = LeakyReLU()(y29)
    y31 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y30)
    y32 = BatchNormalization()(y31)
    y33 = LeakyReLU()(y32)
    y34 = Add()([y33, y30])
    y35 = Conv2D(2*num_channels, (1,1), padding='same', use_bias=False)(y13)
    y36 = BatchNormalization()(y35)
    y37 = LeakyReLU()(y36)
    y38 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y37)
    y39 = BatchNormalization()(y38)
    y40 = LeakyReLU()(y39)
    y41 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y40)
    y42 = BatchNormalization()(y41)
    y43 = LeakyReLU()(y42)
    y44 = Conv2D(2*num_channels, (1,1), padding='same', use_bias=False)(y13)
    y45 = BatchNormalization()(y44)
    y46 = LeakyReLU()(y45)
    y47 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y46)
    y48 = BatchNormalization()(y47)
    y49 = LeakyReLU()(y48)
    y50 = Conv2D(2*num_channels, (1,1), padding='same', use_bias=False)(y49)
    y51 = BatchNormalization()(y50)
    y52 = LeakyReLU()(y51)
    y53 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y52)
    y54 = BatchNormalization()(y53)
    y55 = LeakyReLU()(y54)
    y56 = Concatenate()([y25, y34, y43, y55])
    y57 = Conv2D(16*num_channels, (3,3), (2,2), padding='same', use_bias=False)(y56)
    y58 = BatchNormalization()(y57)
    y59 = LeakyReLU()(y58)
    y60 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y59)
    y61 = BatchNormalization()(y60)
    y62 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y61)
    y63 = BatchNormalization()(y62)
    y64 = LeakyReLU()(y63)
    y65 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y64)
    y66 = BatchNormalization()(y65)
    y67 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y66)
    y68 = BatchNormalization()(y67)
    y69 = LeakyReLU()(y68)
    y70 = Conv2D(16*num_channels, (1,1), padding='same', use_bias=False)(y69)
    y71 = BatchNormalization()(y70)
    y72 = LeakyReLU()(y71)
    y73 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y72)
    y74 = BatchNormalization()(y73)
    y75 = LeakyReLU()(y74)
    y76 = Add()([y69, y75])
    y77 = Conv2D(16*num_channels, (1,1), padding='same', use_bias=False)(y64)
    y78 = BatchNormalization()(y77)
    y79 = LeakyReLU()(y78)
    y80 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y79)
    y81 = BatchNormalization()(y80)
    y82 = LeakyReLU()(y81)
    y83 = Conv2D(16*num_channels, (1,1), padding='same', use_bias=False)(y82)
    y84 = BatchNormalization()(y83)
    y85 = LeakyReLU()(y84)
    y86 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y85)
    y87 = BatchNormalization()(y86)
    y88 = LeakyReLU()(y87)
    y89 = Add()([y82, y88])
    y90 = Concatenate()([y76, y89])
    y91 = GlobalAveragePooling2D()(y90)
    y92 = Flatten()(y91)
    y93 = Dense(10, activation="softmax")(y92)
    y94 =  (y93)
    return Model(inputs=y0, outputs=y94)
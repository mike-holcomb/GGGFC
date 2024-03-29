def build_model_20190717_085746(num_channels):
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
    y16 = Conv2D(1*num_channels, (1,1), padding='same', use_bias=False)(y15)
    y17 = BatchNormalization()(y16)
    y18 = LeakyReLU()(y17)
    y19 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y18)
    y20 = BatchNormalization()(y19)
    y21 = LeakyReLU()(y20)
    y22 = Add()([y21, y15])
    y23 = Concatenate()([y22, y9])
    y24 = MaxPooling2D(2,2,padding='same')(y23)
    y25 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y24)
    y26 = BatchNormalization()(y25)
    y27 = LeakyReLU()(y26)
    y28 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y27)
    y29 = BatchNormalization()(y28)
    y30 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y29)
    y31 = BatchNormalization()(y30)
    y32 = LeakyReLU()(y31)
    y33 = Conv2D(4*num_channels, (1,1), padding='same', use_bias=False)(y27)
    y34 = BatchNormalization()(y33)
    y35 = LeakyReLU()(y34)
    y36 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y35)
    y37 = BatchNormalization()(y36)
    y38 = LeakyReLU()(y37)
    y39 = Concatenate()([y32, y38])
    y40 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y39)
    y41 = BatchNormalization()(y40)
    y42 = LeakyReLU()(y41)
    y43 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y42)
    y44 = BatchNormalization()(y43)
    y45 = LeakyReLU()(y44)
    y46 = Concatenate()([y45, y39])
    y47 = Conv2D(32*num_channels, (3,3), (2,2), padding='same', use_bias=False)(y46)
    y48 = BatchNormalization()(y47)
    y49 = LeakyReLU()(y48)
    y50 = Conv2D(32*num_channels, (1,1), padding='same', use_bias=False)(y49)
    y51 = BatchNormalization()(y50)
    y52 = LeakyReLU()(y51)
    y53 = Conv2D(32*num_channels, (3,3), padding='same', use_bias=False)(y52)
    y54 = BatchNormalization()(y53)
    y55 = LeakyReLU()(y54)
    y56 = Conv2D(32*num_channels, (1,1), padding='same', use_bias=False)(y55)
    y57 = BatchNormalization()(y56)
    y58 = LeakyReLU()(y57)
    y59 = Conv2D(32*num_channels, (3,3), padding='same', use_bias=False)(y58)
    y60 = BatchNormalization()(y59)
    y61 = LeakyReLU()(y60)
    y62 = Conv2D(32*num_channels, (1,1), padding='same', use_bias=False)(y55)
    y63 = BatchNormalization()(y62)
    y64 = LeakyReLU()(y63)
    y65 = Conv2D(32*num_channels, (3,3), padding='same', use_bias=False)(y64)
    y66 = BatchNormalization()(y65)
    y67 = LeakyReLU()(y66)
    y68 = Conv2D(32*num_channels, (1,1), padding='same', use_bias=False)(y55)
    y69 = BatchNormalization()(y68)
    y70 = LeakyReLU()(y69)
    y71 = Conv2D(32*num_channels, (3,3), padding='same', use_bias=False)(y70)
    y72 = BatchNormalization()(y71)
    y73 = LeakyReLU()(y72)
    y74 = Conv2D(32*num_channels, (3,3), padding='same', use_bias=False)(y73)
    y75 = BatchNormalization()(y74)
    y76 = LeakyReLU()(y75)
    y77 = Add()([y73, y76])
    y78 = Conv2D(32*num_channels, (1,1), padding='same', use_bias=False)(y55)
    y79 = BatchNormalization()(y78)
    y80 = LeakyReLU()(y79)
    y81 = Conv2D(32*num_channels, (3,3), padding='same', use_bias=False)(y80)
    y82 = BatchNormalization()(y81)
    y83 = LeakyReLU()(y82)
    y84 = Concatenate()([y61, y67, y77, y83])
    y85 = GlobalMaxPooling2D()(y84)
    y86 = Flatten()(y85)
    y87 = Dense(1024, use_bias=False)(y86)
    y88 = BatchNormalization()(y87)
    y89 = LeakyReLU()(y88)
    y90 = Dense(10, activation="softmax")(y89)
    y91 =  (y90)
    return Model(inputs=y0, outputs=y91)
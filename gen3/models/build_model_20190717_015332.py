def build_model_20190717_015332(num_channels):
    y0 = Input(shape=(32,32,3))
    y1 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y0)
    y2 = BatchNormalization()(y1)
    y3 = LeakyReLU()(y2)
    y4 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y3)
    y5 = BatchNormalization()(y4)
    y6 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y5)
    y7 = BatchNormalization()(y6)
    y8 = LeakyReLU()(y7)
    y9 = Concatenate()([y8, y3])
    y10 = Conv2D(2*num_channels, (1,1), padding='same', use_bias=False)(y9)
    y11 = BatchNormalization()(y10)
    y12 = LeakyReLU()(y11)
    y13 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y12)
    y14 = BatchNormalization()(y13)
    y15 = LeakyReLU()(y14)
    y16 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y15)
    y17 = BatchNormalization()(y16)
    y18 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y17)
    y19 = BatchNormalization()(y18)
    y20 = LeakyReLU()(y19)
    y21 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y9)
    y22 = BatchNormalization()(y21)
    y23 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y22)
    y24 = BatchNormalization()(y23)
    y25 = LeakyReLU()(y24)
    y26 = Conv2D(2*num_channels, (1,1), padding='same', use_bias=False)(y25)
    y27 = BatchNormalization()(y26)
    y28 = LeakyReLU()(y27)
    y29 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y28)
    y30 = BatchNormalization()(y29)
    y31 = LeakyReLU()(y30)
    y32 = Add()([y31, y25])
    y33 = Concatenate()([y20, y32])
    y34 = MaxPooling2D(2,2,padding='same')(y33)
    y35 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y34)
    y36 = BatchNormalization()(y35)
    y37 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y36)
    y38 = BatchNormalization()(y37)
    y39 = LeakyReLU()(y38)
    y40 = Conv2D(8*num_channels, (1,1), padding='same', use_bias=False)(y39)
    y41 = BatchNormalization()(y40)
    y42 = LeakyReLU()(y41)
    y43 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y42)
    y44 = BatchNormalization()(y43)
    y45 = LeakyReLU()(y44)
    y46 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y45)
    y47 = BatchNormalization()(y46)
    y48 = LeakyReLU()(y47)
    y49 = Add()([y45, y48])
    y50 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y39)
    y51 = BatchNormalization()(y50)
    y52 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y51)
    y53 = BatchNormalization()(y52)
    y54 = LeakyReLU()(y53)
    y55 = Conv2D(8*num_channels, (1,1), padding='same', use_bias=False)(y39)
    y56 = BatchNormalization()(y55)
    y57 = LeakyReLU()(y56)
    y58 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y57)
    y59 = BatchNormalization()(y58)
    y60 = LeakyReLU()(y59)
    y61 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y39)
    y62 = BatchNormalization()(y61)
    y63 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y62)
    y64 = BatchNormalization()(y63)
    y65 = LeakyReLU()(y64)
    y66 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y65)
    y67 = BatchNormalization()(y66)
    y68 = LeakyReLU()(y67)
    y69 = Concatenate()([y49, y54, y60, y68])
    y70 = MaxPooling2D(2,2,padding='same')(y69)
    y71 = Conv2D(64*num_channels, (3,3), padding='same', use_bias=False)(y70)
    y72 = BatchNormalization()(y71)
    y73 = LeakyReLU()(y72)
    y74 = Conv2D(64*num_channels, (3,3), padding='same', use_bias=False)(y73)
    y75 = BatchNormalization()(y74)
    y76 = Conv2D(64*num_channels, (3,3), padding='same', use_bias=False)(y75)
    y77 = BatchNormalization()(y76)
    y78 = LeakyReLU()(y77)
    y79 = Conv2D(64*num_channels, (3,3), padding='same', use_bias=False)(y73)
    y80 = BatchNormalization()(y79)
    y81 = LeakyReLU()(y80)
    y82 = Conv2D(64*num_channels, (3,3), padding='same', use_bias=False)(y73)
    y83 = BatchNormalization()(y82)
    y84 = LeakyReLU()(y83)
    y85 = Conv2D(64*num_channels, (3,3), padding='same', use_bias=False)(y73)
    y86 = BatchNormalization()(y85)
    y87 = LeakyReLU()(y86)
    y88 = Concatenate()([y78, y81, y84, y87])
    y89 = GlobalMaxPooling2D()(y88)
    y90 = Flatten()(y89)
    y91 = Dense(10, activation="softmax")(y90)
    y92 =  (y91)
    return Model(inputs=y0, outputs=y92)
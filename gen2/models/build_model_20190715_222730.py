def build_model_20190715_222730(num_channels):
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
    y12 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y11)
    y13 = BatchNormalization()(y12)
    y14 = LeakyReLU()(y13)
    y15 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y6)
    y16 = BatchNormalization()(y15)
    y17 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y16)
    y18 = BatchNormalization()(y17)
    y19 = LeakyReLU()(y18)
    y20 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y6)
    y21 = BatchNormalization()(y20)
    y22 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y21)
    y23 = BatchNormalization()(y22)
    y24 = LeakyReLU()(y23)
    y25 = Concatenate()([y9, y14, y19, y24])
    y26 = Conv2D(4*num_channels, (1,1), padding='same', use_bias=False)(y25)
    y27 = BatchNormalization()(y26)
    y28 = LeakyReLU()(y27)
    y29 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y28)
    y30 = BatchNormalization()(y29)
    y31 = LeakyReLU()(y30)
    y32 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y31)
    y33 = BatchNormalization()(y32)
    y34 = LeakyReLU()(y33)
    y35 = Add()([y34, y31])
    y36 = Conv2D(8*num_channels, (3,3), (2,2), padding='same', use_bias=False)(y35)
    y37 = BatchNormalization()(y36)
    y38 = LeakyReLU()(y37)
    y39 = Conv2D(8*num_channels, (1,1), padding='same', use_bias=False)(y38)
    y40 = BatchNormalization()(y39)
    y41 = LeakyReLU()(y40)
    y42 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y41)
    y43 = BatchNormalization()(y42)
    y44 = LeakyReLU()(y43)
    y45 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y44)
    y46 = BatchNormalization()(y45)
    y47 = LeakyReLU()(y46)
    y48 = Concatenate()([y47, y44])
    y49 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y48)
    y50 = BatchNormalization()(y49)
    y51 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y50)
    y52 = BatchNormalization()(y51)
    y53 = LeakyReLU()(y52)
    y54 = Conv2D(16*num_channels, (1,1), padding='same', use_bias=False)(y53)
    y55 = BatchNormalization()(y54)
    y56 = LeakyReLU()(y55)
    y57 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y56)
    y58 = BatchNormalization()(y57)
    y59 = LeakyReLU()(y58)
    y60 = Add()([y53, y59])
    y61 = Concatenate()([y60, y48])
    y62 = Conv2D(64*num_channels, (3,3), (2,2), padding='same', use_bias=False)(y61)
    y63 = BatchNormalization()(y62)
    y64 = LeakyReLU()(y63)
    y65 = Conv2D(64*num_channels, (1,1), padding='same', use_bias=False)(y64)
    y66 = BatchNormalization()(y65)
    y67 = LeakyReLU()(y66)
    y68 = Conv2D(64*num_channels, (3,3), padding='same', use_bias=False)(y67)
    y69 = BatchNormalization()(y68)
    y70 = LeakyReLU()(y69)
    y71 = Conv2D(64*num_channels, (3,3), padding='same', use_bias=False)(y70)
    y72 = BatchNormalization()(y71)
    y73 = LeakyReLU()(y72)
    y74 = Conv2D(64*num_channels, (1,1), padding='same', use_bias=False)(y73)
    y75 = BatchNormalization()(y74)
    y76 = LeakyReLU()(y75)
    y77 = Conv2D(64*num_channels, (3,3), padding='same', use_bias=False)(y76)
    y78 = BatchNormalization()(y77)
    y79 = LeakyReLU()(y78)
    y80 = GlobalMaxPooling2D()(y79)
    y81 = Flatten()(y80)
    y82 = Dense(1024, use_bias=False)(y81)
    y83 = BatchNormalization()(y82)
    y84 = LeakyReLU()(y83)
    y85 = Dense(10, activation="softmax")(y84)
    y86 =  (y85)
    return Model(inputs=y0, outputs=y86)
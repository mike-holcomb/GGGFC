def build_model_20190716_213242(num_channels):
    y0 = Input(shape=(32,32,3))
    y1 = Conv2D(1*num_channels, (1,1), padding='same', use_bias=False)(y0)
    y2 = BatchNormalization()(y1)
    y3 = LeakyReLU()(y2)
    y4 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y3)
    y5 = BatchNormalization()(y4)
    y6 = LeakyReLU()(y5)
    y7 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y6)
    y8 = BatchNormalization()(y7)
    y9 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y8)
    y10 = BatchNormalization()(y9)
    y11 = LeakyReLU()(y10)
    y12 = Conv2D(1*num_channels, (1,1), padding='same', use_bias=False)(y6)
    y13 = BatchNormalization()(y12)
    y14 = LeakyReLU()(y13)
    y15 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y14)
    y16 = BatchNormalization()(y15)
    y17 = LeakyReLU()(y16)
    y18 = Concatenate()([y11, y17])
    y19 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y18)
    y20 = BatchNormalization()(y19)
    y21 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y20)
    y22 = BatchNormalization()(y21)
    y23 = LeakyReLU()(y22)
    y24 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y23)
    y25 = BatchNormalization()(y24)
    y26 = LeakyReLU()(y25)
    y27 = Add()([y26, y23])
    y28 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y18)
    y29 = BatchNormalization()(y28)
    y30 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y29)
    y31 = BatchNormalization()(y30)
    y32 = LeakyReLU()(y31)
    y33 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y32)
    y34 = BatchNormalization()(y33)
    y35 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y34)
    y36 = BatchNormalization()(y35)
    y37 = LeakyReLU()(y36)
    y38 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y18)
    y39 = BatchNormalization()(y38)
    y40 = LeakyReLU()(y39)
    y41 = Conv2D(2*num_channels, (1,1), padding='same', use_bias=False)(y40)
    y42 = BatchNormalization()(y41)
    y43 = LeakyReLU()(y42)
    y44 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y43)
    y45 = BatchNormalization()(y44)
    y46 = LeakyReLU()(y45)
    y47 = Add()([y46, y40])
    y48 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y18)
    y49 = BatchNormalization()(y48)
    y50 = LeakyReLU()(y49)
    y51 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y50)
    y52 = BatchNormalization()(y51)
    y53 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y52)
    y54 = BatchNormalization()(y53)
    y55 = LeakyReLU()(y54)
    y56 = Add()([y55, y50])
    y57 = Concatenate()([y27, y37, y47, y56])
    y58 = Conv2D(16*num_channels, (3,3), (2,2), padding='same', use_bias=False)(y57)
    y59 = BatchNormalization()(y58)
    y60 = LeakyReLU()(y59)
    y61 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y60)
    y62 = BatchNormalization()(y61)
    y63 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y62)
    y64 = BatchNormalization()(y63)
    y65 = LeakyReLU()(y64)
    y66 = Conv2D(16*num_channels, (1,1), padding='same', use_bias=False)(y65)
    y67 = BatchNormalization()(y66)
    y68 = LeakyReLU()(y67)
    y69 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y68)
    y70 = BatchNormalization()(y69)
    y71 = LeakyReLU()(y70)
    y72 = Conv2D(16*num_channels, (1,1), padding='same', use_bias=False)(y65)
    y73 = BatchNormalization()(y72)
    y74 = LeakyReLU()(y73)
    y75 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y74)
    y76 = BatchNormalization()(y75)
    y77 = LeakyReLU()(y76)
    y78 = Concatenate()([y71, y77])
    y79 = MaxPooling2D(2,2,padding='same')(y78)
    y80 = Conv2D(64*num_channels, (1,1), padding='same', use_bias=False)(y79)
    y81 = BatchNormalization()(y80)
    y82 = LeakyReLU()(y81)
    y83 = Conv2D(64*num_channels, (3,3), padding='same', use_bias=False)(y82)
    y84 = BatchNormalization()(y83)
    y85 = LeakyReLU()(y84)
    y86 = Conv2D(64*num_channels, (3,3), padding='same', use_bias=False)(y85)
    y87 = BatchNormalization()(y86)
    y88 = LeakyReLU()(y87)
    y89 = GlobalAveragePooling2D()(y88)
    y90 = Flatten()(y89)
    y91 = Dense(1024, use_bias=False)(y90)
    y92 = BatchNormalization()(y91)
    y93 = LeakyReLU()(y92)
    y94 = Dense(10, activation="softmax")(y93)
    y95 =  (y94)
    return Model(inputs=y0, outputs=y95)
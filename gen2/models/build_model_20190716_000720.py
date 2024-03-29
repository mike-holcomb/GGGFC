def build_model_20190716_000720(num_channels):
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
    y12 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y6)
    y13 = BatchNormalization()(y12)
    y14 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y13)
    y15 = BatchNormalization()(y14)
    y16 = LeakyReLU()(y15)
    y17 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y16)
    y18 = BatchNormalization()(y17)
    y19 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y18)
    y20 = BatchNormalization()(y19)
    y21 = LeakyReLU()(y20)
    y22 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y6)
    y23 = BatchNormalization()(y22)
    y24 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y23)
    y25 = BatchNormalization()(y24)
    y26 = LeakyReLU()(y25)
    y27 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y26)
    y28 = BatchNormalization()(y27)
    y29 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y28)
    y30 = BatchNormalization()(y29)
    y31 = LeakyReLU()(y30)
    y32 = Add()([y26, y31])
    y33 = Conv2D(1*num_channels, (1,1), padding='same', use_bias=False)(y6)
    y34 = BatchNormalization()(y33)
    y35 = LeakyReLU()(y34)
    y36 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y35)
    y37 = BatchNormalization()(y36)
    y38 = LeakyReLU()(y37)
    y39 = Conv2D(1*num_channels, (1,1), padding='same', use_bias=False)(y38)
    y40 = BatchNormalization()(y39)
    y41 = LeakyReLU()(y40)
    y42 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y41)
    y43 = BatchNormalization()(y42)
    y44 = LeakyReLU()(y43)
    y45 = Add()([y38, y44])
    y46 = Concatenate()([y11, y21, y32, y45])
    y47 = MaxPooling2D(2,2,padding='same')(y46)
    y48 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y47)
    y49 = BatchNormalization()(y48)
    y50 = LeakyReLU()(y49)
    y51 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y50)
    y52 = BatchNormalization()(y51)
    y53 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y52)
    y54 = BatchNormalization()(y53)
    y55 = LeakyReLU()(y54)
    y56 = MaxPooling2D(2,2,padding='same')(y55)
    y57 = Conv2D(16*num_channels, (1,1), padding='same', use_bias=False)(y56)
    y58 = BatchNormalization()(y57)
    y59 = LeakyReLU()(y58)
    y60 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y59)
    y61 = BatchNormalization()(y60)
    y62 = LeakyReLU()(y61)
    y63 = Conv2D(16*num_channels, (1,1), padding='same', use_bias=False)(y62)
    y64 = BatchNormalization()(y63)
    y65 = LeakyReLU()(y64)
    y66 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y65)
    y67 = BatchNormalization()(y66)
    y68 = LeakyReLU()(y67)
    y69 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y62)
    y70 = BatchNormalization()(y69)
    y71 = LeakyReLU()(y70)
    y72 = Conv2D(16*num_channels, (1,1), padding='same', use_bias=False)(y62)
    y73 = BatchNormalization()(y72)
    y74 = LeakyReLU()(y73)
    y75 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y74)
    y76 = BatchNormalization()(y75)
    y77 = LeakyReLU()(y76)
    y78 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y62)
    y79 = BatchNormalization()(y78)
    y80 = LeakyReLU()(y79)
    y81 = Concatenate()([y68, y71, y77, y80])
    y82 = MaxPooling2D(2,2,padding='same')(y81)
    y83 = Conv2D(128*num_channels, (1,1), padding='same', use_bias=False)(y82)
    y84 = BatchNormalization()(y83)
    y85 = LeakyReLU()(y84)
    y86 = Conv2D(128*num_channels, (3,3), padding='same', use_bias=False)(y85)
    y87 = BatchNormalization()(y86)
    y88 = LeakyReLU()(y87)
    y89 = Conv2D(128*num_channels, (3,3), padding='same', use_bias=False)(y88)
    y90 = BatchNormalization()(y89)
    y91 = Conv2D(128*num_channels, (3,3), padding='same', use_bias=False)(y90)
    y92 = BatchNormalization()(y91)
    y93 = LeakyReLU()(y92)
    y94 = GlobalMaxPooling2D()(y93)
    y95 = Flatten()(y94)
    y96 = Dense(10, activation="softmax")(y95)
    y97 =  (y96)
    return Model(inputs=y0, outputs=y97)
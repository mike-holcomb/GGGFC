def build_model_20190717_091913(num_channels):
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
    y12 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y11)
    y13 = BatchNormalization()(y12)
    y14 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y13)
    y15 = BatchNormalization()(y14)
    y16 = LeakyReLU()(y15)
    y17 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y16)
    y18 = BatchNormalization()(y17)
    y19 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y18)
    y20 = BatchNormalization()(y19)
    y21 = LeakyReLU()(y20)
    y22 = Add()([y16, y21])
    y23 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y11)
    y24 = BatchNormalization()(y23)
    y25 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y24)
    y26 = BatchNormalization()(y25)
    y27 = LeakyReLU()(y26)
    y28 = Conv2D(1*num_channels, (1,1), padding='same', use_bias=False)(y27)
    y29 = BatchNormalization()(y28)
    y30 = LeakyReLU()(y29)
    y31 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y30)
    y32 = BatchNormalization()(y31)
    y33 = LeakyReLU()(y32)
    y34 = Concatenate()([y22, y33])
    y35 = MaxPooling2D(2,2,padding='same')(y34)
    y36 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y35)
    y37 = BatchNormalization()(y36)
    y38 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y37)
    y39 = BatchNormalization()(y38)
    y40 = LeakyReLU()(y39)
    y41 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y40)
    y42 = BatchNormalization()(y41)
    y43 = LeakyReLU()(y42)
    y44 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y43)
    y45 = BatchNormalization()(y44)
    y46 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y45)
    y47 = BatchNormalization()(y46)
    y48 = LeakyReLU()(y47)
    y49 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y40)
    y50 = BatchNormalization()(y49)
    y51 = LeakyReLU()(y50)
    y52 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y51)
    y53 = BatchNormalization()(y52)
    y54 = LeakyReLU()(y53)
    y55 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y40)
    y56 = BatchNormalization()(y55)
    y57 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y56)
    y58 = BatchNormalization()(y57)
    y59 = LeakyReLU()(y58)
    y60 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y59)
    y61 = BatchNormalization()(y60)
    y62 = LeakyReLU()(y61)
    y63 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y40)
    y64 = BatchNormalization()(y63)
    y65 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y64)
    y66 = BatchNormalization()(y65)
    y67 = LeakyReLU()(y66)
    y68 = Conv2D(4*num_channels, (1,1), padding='same', use_bias=False)(y67)
    y69 = BatchNormalization()(y68)
    y70 = LeakyReLU()(y69)
    y71 = Conv2D(4*num_channels, (3,3), padding='same', use_bias=False)(y70)
    y72 = BatchNormalization()(y71)
    y73 = LeakyReLU()(y72)
    y74 = Add()([y67, y73])
    y75 = Concatenate()([y48, y54, y62, y74])
    y76 = GlobalAveragePooling2D()(y75)
    y77 = Flatten()(y76)
    y78 = Dense(1024, use_bias=False)(y77)
    y79 = BatchNormalization()(y78)
    y80 = LeakyReLU()(y79)
    y81 = Dense(10, activation="softmax")(y80)
    y82 =  (y81)
    return Model(inputs=y0, outputs=y82)
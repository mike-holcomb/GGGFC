def build_model_20190716_003436(num_channels):
    y0 = Input(shape=(32,32,3))
    y1 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y0)
    y2 = BatchNormalization()(y1)
    y3 = LeakyReLU()(y2)
    y4 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y3)
    y5 = BatchNormalization()(y4)
    y6 = LeakyReLU()(y5)
    y7 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y3)
    y8 = BatchNormalization()(y7)
    y9 = LeakyReLU()(y8)
    y10 = Concatenate()([y6, y9])
    y11 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y10)
    y12 = BatchNormalization()(y11)
    y13 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y12)
    y14 = BatchNormalization()(y13)
    y15 = LeakyReLU()(y14)
    y16 = Conv2D(2*num_channels, (1,1), padding='same', use_bias=False)(y15)
    y17 = BatchNormalization()(y16)
    y18 = LeakyReLU()(y17)
    y19 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y18)
    y20 = BatchNormalization()(y19)
    y21 = LeakyReLU()(y20)
    y22 = Add()([y21, y15])
    y23 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y10)
    y24 = BatchNormalization()(y23)
    y25 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y24)
    y26 = BatchNormalization()(y25)
    y27 = LeakyReLU()(y26)
    y28 = Conv2D(2*num_channels, (1,1), padding='same', use_bias=False)(y27)
    y29 = BatchNormalization()(y28)
    y30 = LeakyReLU()(y29)
    y31 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y30)
    y32 = BatchNormalization()(y31)
    y33 = LeakyReLU()(y32)
    y34 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y10)
    y35 = BatchNormalization()(y34)
    y36 = LeakyReLU()(y35)
    y37 = Conv2D(2*num_channels, (1,1), padding='same', use_bias=False)(y36)
    y38 = BatchNormalization()(y37)
    y39 = LeakyReLU()(y38)
    y40 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y39)
    y41 = BatchNormalization()(y40)
    y42 = LeakyReLU()(y41)
    y43 = Add()([y42, y36])
    y44 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y10)
    y45 = BatchNormalization()(y44)
    y46 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y45)
    y47 = BatchNormalization()(y46)
    y48 = LeakyReLU()(y47)
    y49 = Conv2D(2*num_channels, (1,1), padding='same', use_bias=False)(y48)
    y50 = BatchNormalization()(y49)
    y51 = LeakyReLU()(y50)
    y52 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y51)
    y53 = BatchNormalization()(y52)
    y54 = LeakyReLU()(y53)
    y55 = Concatenate()([y22, y33, y43, y54])
    y56 = MaxPooling2D(2,2,padding='same')(y55)
    y57 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y56)
    y58 = BatchNormalization()(y57)
    y59 = LeakyReLU()(y58)
    y60 = Conv2D(16*num_channels, (1,1), padding='same', use_bias=False)(y59)
    y61 = BatchNormalization()(y60)
    y62 = LeakyReLU()(y61)
    y63 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y62)
    y64 = BatchNormalization()(y63)
    y65 = LeakyReLU()(y64)
    y66 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y59)
    y67 = BatchNormalization()(y66)
    y68 = LeakyReLU()(y67)
    y69 = Concatenate()([y65, y68])
    y70 = Conv2D(32*num_channels, (3,3), padding='same', use_bias=False)(y69)
    y71 = BatchNormalization()(y70)
    y72 = Conv2D(32*num_channels, (3,3), padding='same', use_bias=False)(y71)
    y73 = BatchNormalization()(y72)
    y74 = LeakyReLU()(y73)
    y75 = Conv2D(32*num_channels, (3,3), padding='same', use_bias=False)(y74)
    y76 = BatchNormalization()(y75)
    y77 = LeakyReLU()(y76)
    y78 = Add()([y74, y77])
    y79 = GlobalMaxPooling2D()(y78)
    y80 = Flatten()(y79)
    y81 = Dense(1024, use_bias=False)(y80)
    y82 = BatchNormalization()(y81)
    y83 = LeakyReLU()(y82)
    y84 = Dense(10, activation="softmax")(y83)
    y85 =  (y84)
    return Model(inputs=y0, outputs=y85)
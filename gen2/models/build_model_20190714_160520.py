def build_model_20190714_160520(num_channels):
    y0 = Input(shape=(32,32,3))
    y1 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y0)
    y2 = BatchNormalization()(y1)
    y3 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y2)
    y4 = BatchNormalization()(y3)
    y5 = LeakyReLU()(y4)
    y6 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y5)
    y7 = BatchNormalization()(y6)
    y8 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y7)
    y9 = BatchNormalization()(y8)
    y10 = LeakyReLU()(y9)
    y11 = Conv2D(1*num_channels, (1,1), padding='same', use_bias=False)(y5)
    y12 = BatchNormalization()(y11)
    y13 = LeakyReLU()(y12)
    y14 = Conv2D(1*num_channels, (3,3), padding='same', use_bias=False)(y13)
    y15 = BatchNormalization()(y14)
    y16 = LeakyReLU()(y15)
    y17 = Concatenate()([y10, y16])
    y18 = Conv2D(2*num_channels, (1,1), padding='same', use_bias=False)(y17)
    y19 = BatchNormalization()(y18)
    y20 = LeakyReLU()(y19)
    y21 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y20)
    y22 = BatchNormalization()(y21)
    y23 = LeakyReLU()(y22)
    y24 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y23)
    y25 = BatchNormalization()(y24)
    y26 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y25)
    y27 = BatchNormalization()(y26)
    y28 = LeakyReLU()(y27)
    y29 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y17)
    y30 = BatchNormalization()(y29)
    y31 = LeakyReLU()(y30)
    y32 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y31)
    y33 = BatchNormalization()(y32)
    y34 = LeakyReLU()(y33)
    y35 = Add()([y34, y31])
    y36 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y35)
    y37 = BatchNormalization()(y36)
    y38 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y37)
    y39 = BatchNormalization()(y38)
    y40 = LeakyReLU()(y39)
    y41 = Conv2D(2*num_channels, (3,3), padding='same', use_bias=False)(y40)
    y42 = BatchNormalization()(y41)
    y43 = LeakyReLU()(y42)
    y44 = Add()([y40, y43])
    y45 = Add()([y35, y44])
    y46 = Concatenate()([y28, y45])
    y47 = Conv2D(8*num_channels, (3,3), (2,2), padding='same', use_bias=False)(y46)
    y48 = BatchNormalization()(y47)
    y49 = LeakyReLU()(y48)
    y50 = Conv2D(8*num_channels, (1,1), padding='same', use_bias=False)(y49)
    y51 = BatchNormalization()(y50)
    y52 = LeakyReLU()(y51)
    y53 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y52)
    y54 = BatchNormalization()(y53)
    y55 = LeakyReLU()(y54)
    y56 = Conv2D(8*num_channels, (1,1), padding='same', use_bias=False)(y55)
    y57 = BatchNormalization()(y56)
    y58 = LeakyReLU()(y57)
    y59 = Conv2D(8*num_channels, (3,3), padding='same', use_bias=False)(y58)
    y60 = BatchNormalization()(y59)
    y61 = LeakyReLU()(y60)
    y62 = Concatenate()([y55, y61])
    y63 = Conv2D(16*num_channels, (1,1), padding='same', use_bias=False)(y62)
    y64 = BatchNormalization()(y63)
    y65 = LeakyReLU()(y64)
    y66 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y65)
    y67 = BatchNormalization()(y66)
    y68 = LeakyReLU()(y67)
    y69 = Conv2D(16*num_channels, (3,3), padding='same', use_bias=False)(y68)
    y70 = BatchNormalization()(y69)
    y71 = LeakyReLU()(y70)
    y72 = Add()([y71, y68])
    y73 = Concatenate()([y62, y72])
    y74 = GlobalMaxPooling2D()(y73)
    y75 = Flatten()(y74)
    y76 = Dense(10, activation="softmax")(y75)
    y77 =  (y76)
    return Model(inputs=y0, outputs=y77)
import keras
import tensorflow

def UNet_1():
    """
        Basic Model from article https://arxiv.org/abs/1505.04597
        Detailed code
    """
    Input = keras.layers.Input( shape=(572,572,1) )     # shape = (572,572,1)

    # Contraction/Downsampling path
    Conv1 = keras.layers.Conv2D(filters = 64, kernel_size = (3,3), padding = 'valid', activation='relu')(Input)        # shape = (570,570,64)
    Conv1 = keras.layers.Conv2D(filters = 64, kernel_size = (3,3), padding = 'valid', activation='relu')(Conv1)        # shape = (568,568,64)
    MaxPool1 = keras.layers.MaxPooling2D(pool_size = (2, 2), padding='valid')(Conv1)                # shape = (284,284,64)

    Conv2 = keras.layers.Conv2D(filters = 128, kernel_size = (3,3), padding = 'valid', activation='relu')(MaxPool1)        # shape = (282,282,128)
    Conv2 = keras.layers.Conv2D(filters = 128, kernel_size = (3,3), padding = 'valid', activation='relu')(Conv2)        # shape = (280,280,128)
    MaxPool2 = keras.layers.MaxPooling2D(pool_size = (2, 2), padding='valid')(Conv2)                # shape = (140,140,128)

    Conv3 = keras.layers.Conv2D(filters = 256, kernel_size = (3,3), padding = 'valid', activation='relu')(MaxPool2)        # shape = (138,138,256)
    Conv3 = keras.layers.Conv2D(filters = 256, kernel_size = (3,3), padding = 'valid', activation='relu')(Conv3)        # shape = (136,136,256)
    MaxPool3 = keras.layers.MaxPooling2D(pool_size = (2, 2), padding='valid')(Conv3)                # shape = (68,68,256)

    Conv4 = keras.layers.Conv2D(filters = 512, kernel_size = (3,3), padding = 'valid', activation='relu')(MaxPool3)        # shape = (66,66,512)
    Conv4 = keras.layers.Conv2D(filters = 512, kernel_size = (3,3), padding = 'valid', activation='relu')(Conv4)        # shape = (64,64,512)
    MaxPool4 = keras.layers.MaxPooling2D(pool_size = (2, 2), padding='valid')(Conv4)                # shape = (32,32,512)

    # Bottom
    Bottom = keras.layers.Conv2D(filters = 1024, kernel_size = (3,3), padding = 'valid', activation='relu')(MaxPool4)        # shape = (30,30,1024)
    Bottom = keras.layers.Conv2D(filters = 1024, kernel_size = (3,3), padding = 'valid', activation='relu')(Bottom)        # shape = (28,28,1024)

    # Epensive/Upsampling path
    Conv4Crop = keras.layers.Cropping2D(cropping=((4, 4), (4, 4)))(Conv4)                                                   # shape = (56,56,512)
    UpConv4 = keras.layers.Conv2DTranspose(filters = 512, kernel_size = (2,2), strides=(2, 2), padding='same', activation='relu')(Bottom)  # shape = (56,56,512)
    UpConv4 = keras.layers.Concatenate(axis=-1)([Conv4Crop, UpConv4])                                                                # shape = (56,56,1024)
    UpConv4 = keras.layers.Conv2D(filters = 512, kernel_size = (3,3), padding = 'valid', activation='relu')(UpConv4)   # shape = (54,54,512)
    UpConv4 = keras.layers.Conv2D(filters = 512, kernel_size = (3,3), padding = 'valid', activation='relu')(UpConv4)   # shape = (52,52,512)
    
    Conv3Crop = keras.layers.Cropping2D(cropping=((16, 16), (16, 16)))(Conv3)                                                   # shape = (104,104,256)
    UpConv3 = keras.layers.Conv2DTranspose(filters = 256, kernel_size = (2,2), strides=(2, 2), padding = 'same', activation='relu')(UpConv4)  # shape = (104,104,256)
    UpConv3 = keras.layers.Concatenate(axis=-1)([Conv3Crop, UpConv3])                                                                # shape = (104,104,512)
    UpConv3 = keras.layers.Conv2D(filters = 256, kernel_size = (3,3), padding = 'valid', activation='relu')(UpConv3)   # shape = (102,102,256)
    UpConv3 = keras.layers.Conv2D(filters = 256, kernel_size = (3,3), padding = 'valid', activation='relu')(UpConv3)   # shape = (100,100,256)

    Conv2Crop = keras.layers.Cropping2D(cropping=((40, 40), (40, 40)))(Conv2)                                               # shape = (200,200,128)
    UpConv2 = keras.layers.Conv2DTranspose(filters = 128, kernel_size = (2,2), strides=(2, 2), padding = 'same', activation='relu')(UpConv3)  # shape = (200,200,128)
    UpConv2 = keras.layers.Concatenate(axis=-1)([Conv2Crop, UpConv2])                                                        # shape = (200,200,256)
    UpConv2 = keras.layers.Conv2D(filters = 128, kernel_size = (3,3), padding = 'valid', activation='relu')(UpConv2)   # shape = (198,198,128)
    UpConv2 = keras.layers.Conv2D(filters = 128, kernel_size = (3,3), padding = 'valid', activation='relu')(UpConv2)   # shape = (196,196,128)

    Conv1Crop = keras.layers.Cropping2D(cropping=((88, 88), (88, 88)))(Conv1)                                               # shape = (392,392,64)
    UpConv1 = keras.layers.Conv2DTranspose(filters = 64, kernel_size = (2,2), strides=(2, 2), padding = 'same', activation='relu')(UpConv2)  # shape = (392,392,64)
    UpConv1 = keras.layers.Concatenate(axis=-1)([Conv1Crop, UpConv1])                                                        # shape = (392,392,128)
    UpConv1 = keras.layers.Conv2D(filters = 64, kernel_size = (3,3), padding = 'valid', activation='relu')(UpConv1)   # shape = (390,390,64)
    UpConv1 = keras.layers.Conv2D(filters = 64, kernel_size = (3,3), padding = 'valid', activation='relu')(UpConv1)   # shape = (388,388,64)

    Output = keras.layers.Dense(units = 2, activation='softmax')(UpConv1)   # shape = (388,388,2)

    return keras.models.Model( Input, Output )

def UNet_2():
    """
        Simplified code
    """

    def ConvolutionBlock(filters, inputs):
        y = keras.layers.Conv2D(filters = filters, kernel_size = (3,3), padding = 'valid', activation='relu')(inputs)
        y = keras.layers.Conv2D(filters = filters, kernel_size = (3,3), padding = 'valid', activation='relu')(y)
        return y

    def MaxPool(inputs):
        return keras.layers.MaxPooling2D(pool_size = (2, 2), padding='valid')(inputs)

    def UpConvolution( UpConv, Conv ):
        crop_x = int( (Conv.shape[1] - 2*UpConv.shape[1])/2 )   # Dimension of cropping
        crop_y = int( (Conv.shape[2] - 2*UpConv.shape[2])/2 )   # Dimension of cropping
        filters = Conv.shape[3]                                 # number of channels

        Crop = keras.layers.Cropping2D(cropping=((crop_x, crop_x), (crop_y, crop_y)) )(Conv)
        Out = keras.layers.Conv2DTranspose(filters = filters, kernel_size = (2,2), strides=(2, 2), padding='same', activation='relu')(UpConv)
        Out = keras.layers.Concatenate(axis=-1)([Crop, Out])
        Out = keras.layers.Conv2D(filters = filters, kernel_size = (3,3), padding = 'valid', activation='relu')(Out)
        Out = keras.layers.Conv2D(filters = filters, kernel_size = (3,3), padding = 'valid', activation='relu')(Out)
        return Out

    Input = keras.layers.Input( shape=(572,572,1) )     # shape = (572,572,1)

    # Contraction/Downsampling path
    Conv1 = ConvolutionBlock(64, Input)                  # shape = (568,568,64)
    Conv2 = ConvolutionBlock(128, MaxPool(Conv1))        # shape = (282,282,128)
    Conv3 = ConvolutionBlock(256, MaxPool(Conv2))       # shape = (136,136,256)
    Conv4 = ConvolutionBlock(512, MaxPool(Conv3))        # shape = (64,64,512)
    
    # Bottom
    Bottom = ConvolutionBlock(512, MaxPool(Conv4))        # shape = (28,28,1024)

    # Epensive/Upsampling path
    UpConv4 = UpConvolution( Bottom, Conv4 )   # shape = (52,52,512)
    UpConv3 = UpConvolution( UpConv4, Conv3 )   # shape = (100,100,256)                                         # shape = (200,200,128)
    UpConv2 = UpConvolution( UpConv3, Conv2 )   # shape = (196,196,128)
    UpConv1 = UpConvolution( UpConv2, Conv1 )   # shape = (388,388,64)

    Output = keras.layers.Dense(units = 2, activation='softmax')(UpConv1)   # shape = (388,388,2)

    return keras.models.Model( Input, Output )



def UNet_3():
    """
        Unet version with resizing, so the output dimension is equal to the input dimension
    """

    def ConvolutionBlock(filters, inputs):
        """
            padding = 'same' will not change image dimension
        """
        y = keras.layers.Conv2D(filters = filters, kernel_size = (3,3), padding = 'same', activation='relu')(inputs)
        y = keras.layers.Conv2D(filters = filters, kernel_size = (3,3), padding = 'same', activation='relu')(y)
        return y

    def MaxPool(inputs):
        return keras.layers.MaxPooling2D(pool_size = (2, 2), padding='valid')(inputs)

    def UpConvolution( UpConv, Conv ):
        def resize_like(input_tensor, ref_tensor): # resizes input tensor wrt. ref_tensor   # https://stackoverflow.com/questions/46418373
            H, W = ref_tensor.get_shape()[1], ref_tensor.get_shape()[2]
            return tensorflow.image.resize(input_tensor, [H, W])
        
        filters = Conv.shape[3]                                 # number of channels

        
        Out = keras.layers.Conv2DTranspose(filters = filters, kernel_size = (2,2), strides=(2, 2), padding='same', activation='relu')(UpConv)
        Out = resize_like(Out, Conv)            # resize Out to be the same dimension of Conv
        Out = keras.layers.Concatenate(axis=-1)([Conv, Out])
        Out = keras.layers.Conv2D(filters = filters, kernel_size = (3,3), padding = 'same', activation='relu')(Out)
        Out = keras.layers.Conv2D(filters = filters, kernel_size = (3,3), padding = 'same', activation='relu')(Out)
        return Out

    Input = keras.layers.Input( shape=(572,572,1) )     # shape = (572,572,1)

    # Contraction/Downsampling path
    Conv1 = ConvolutionBlock(64, Input)                  # shape = (572,572,64)
    Conv2 = ConvolutionBlock(128, MaxPool(Conv1))        # shape = (286,286,128)
    Conv3 = ConvolutionBlock(256, MaxPool(Conv2))        # shape = (143,143,256)
    Conv4 = ConvolutionBlock(512, MaxPool(Conv3))        # shape = (71,71,512)
    
    # Bottom
    Bottom = ConvolutionBlock(512, MaxPool(Conv4))        # shape = (35,35,1024)

    # Epensive/Upsampling path
    UpConv4 = UpConvolution( Bottom, Conv4 )   # shape = (71,71,512)
    UpConv3 = UpConvolution( UpConv4, Conv3 )   # shape = (143,143,256)                                         # shape = (200,200,128)
    UpConv2 = UpConvolution( UpConv3, Conv2 )   # shape = (286,286,128)
    UpConv1 = UpConvolution( UpConv2, Conv1 )   # shape = (572,572,64)

    Output = keras.layers.Dense(units = 2, activation='softmax')(UpConv1)   # shape = (572,572,2)

    return keras.models.Model( Input, Output )



####

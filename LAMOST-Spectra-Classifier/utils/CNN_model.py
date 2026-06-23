from tensorflow.keras import layers, models

def CNN_Model_1D(input_shape=500, output_shape=3, conv_branch_num=3, 
                 conv_kernel_size=[3, 5, 7, 9, 11], node_num=[128, 64, 32]):
    
    '''
    Build a 1D Convolutional Neural Network (CNN) model for spectral classification.

    Parameters
    ----------
    input_shape : int
        The shape of the input data. The default is 500.
    output_shape : int
        The number of output classes. The default is 3.
    conv_branch_num : int
        The number of convolutional branches. The default is 3.
    conv_kernel_size : list
        The kernel sizes for the convolutional layers. The default is [3, 5, 7, 9, 11].
    node_num : list
        The number of nodes in the fully connected layers. The default is [128, 64, 32].
    
    Returns
    -------
    model : keras.Model
        The 1D CNN model built.
    '''

    # input layer
    input_layer = layers.Input(shape=(input_shape, 1))

    # convolution branches & layers
    convolution_branches = []
    for kernel_size in conv_kernel_size:
        my_layers = input_layer
        for branch_number in range (conv_branch_num):
            my_layers = layers.Conv1D(filters=32,
                                      kernel_size=kernel_size,
                                      strides=1,
                                      padding='same',
                                      activation='relu',
                                      use_bias=False,
                                      )(my_layers)
            my_layers = layers.BatchNormalization()(my_layers)
            my_layers = layers.MaxPool1D(pool_size=3, strides=3)(my_layers)
        convolution_branches.append(my_layers)
    my_layers = layers.concatenate(convolution_branches, axis=-1)
    my_layers = layers.Flatten()(my_layers)

    # fully connected dense layers
    for node_number in node_num:
        my_layers = layers.Dense(units=node_number, activation='relu', use_bias=False)(my_layers)
        my_layers = layers.BatchNormalization()(my_layers)

    # output layer
    my_layers = layers.Dense(units=output_shape, activation='softmax', use_bias=False)(my_layers)

    return models.Model(inputs=input_layer, outputs=my_layers, name='CNN_Model_1D')

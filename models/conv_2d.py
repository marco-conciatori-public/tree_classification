import torch
from torch import nn


class Conv_2d(nn.Module):
    def __init__(
            self,
            input_shape,
            num_output: int,
            num_conv_layers: int,
            dense_layers: list,
            convolution_parameters: dict,
            pooling_operation: str = None,
            pooling_parameters: dict = None,
            name: str = None,
            model_id: int = 0,
    ):
        super(Conv_2d, self).__init__()
        self.input_shape = input_shape
        self.num_output = num_output
        self.name = name
        self.id = model_id

        convolutional_layer_list = []
        for i in range(num_conv_layers):
            # convolution_parameters contains:
            #   - in_channels: int,
            #   - out_channels: int,
            #   - groups: int,
            #   - kernel_size: int,
            #   - stride: int,
            #   - padding: int,

            # Convolutional layer
            convolutional_layer_list.append(
                nn.Conv2d(**convolution_parameters)
            )
            # Batch normalization layer
            convolutional_layer_list.append(
                nn.BatchNorm2d(convolution_parameters['out_channels'])
            )
            # ReLU layer
            convolutional_layer_list.append(
                nn.ReLU()
            )
            # Pooling layer
            if pooling_operation is not None:
                # pooling_parameters contains:
                #   - kernel_size: int,
                #   - stride: int,
                #   - padding: int,
                convolutional_layer_list.append(
                    getattr(nn, pooling_operation)(**pooling_parameters)
                )

            # update the number of input and output channels for the next conv layer
            convolution_parameters['in_channels'] = convolution_parameters['out_channels']
            convolution_parameters['out_channels'] *= 2

        layers_dimension = []
        # TODO: substitute 6400 with the correct number
        layers_dimension.append(6400)
        layers_dimension.extend(dense_layers)
        layers_dimension.append(self.num_output)
        dense_layer_list = []
        for layer_index in range(len(layers_dimension) - 1):
            # Dense layer
            dense_layer_list.append(
                nn.Linear(
                    in_features=layers_dimension[layer_index],
                    out_features=layers_dimension[layer_index + 1]
                )
            )
            # ReLU layer
            dense_layer_list.append(
                nn.ReLU()
            )

        self.layers = nn.ModuleList(convolutional_layer_list)
        self.layers.append(nn.Flatten())
        self.layers.extend(dense_layer_list)
        # substitute the last ReLU layer with a Softmax
        # self.layers[-1] = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor):
        # print(f'Input shape: {x.shape}')
        counter = 0
        for layer in self.layers:
            x = layer(x)
            # print(f'{counter}° layer ({layer.__class__.__name__}): {x.shape}')
            counter += 1

        return x

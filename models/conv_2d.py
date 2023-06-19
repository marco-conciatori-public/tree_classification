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
        self.pooling_operation = pooling_operation
        self.name = name
        self.id = model_id

        layers_dimension = []
        layers_dimension.append(2)
        layers_dimension.extend(dense_layers)
        layers_dimension.append(self.num_output)

        # convolution_parameters contains:
        #   - kernel_size: int,
        #   - stride: int,
        #   - padding: int,
        convolution_parameters['in_channels'] =
        convolution_parameters['out_channels'] = self.input_shape[self.features_position]  # number of features
        convolution_parameters['groups'] = 1
        # pooling_parameters contains:
        #   - kernel_size: int,
        #   - stride: int,
        #   - padding: int,
        if self.pooling_operation is not None:
            # self.generic_pooling = nn.MaxPool1d(**pooling_parameters)
            self.generic_pooling = getattr(nn, self.pooling_operation)(**pooling_parameters)

        convolutional_layer_list = []
        for i in range(num_conv_layers):
            convolutional_layer_list.append(
                nn.Conv1d(**convolution_parameters)
            )
        self.convolutional_layers = nn.ModuleList(convolutional_layer_list)

        dense_layer_list = []
        for layer_index in range(len(layers_dimension) - 1):
            dense_layer_list.append(
                nn.Linear(
                    in_features=layers_dimension[layer_index],
                    out_features=layers_dimension[layer_index + 1]
                )
            )
        self.dense_layers = nn.ModuleList(dense_layer_list)

        # reshape input from (batch, flattened features, convolutional output)
        #   to (batch, flattened features * convolutional output)
        # start_dim=-len(input_dimensions) works even for not batched input
        self.flatten_input = nn.Flatten(start_dim=-len(self.input_shape), end_dim=-1)

    def forward(self, x: torch.Tensor):
        # x: tensor of shape (number of past days, flattened features)
        # flattened features is the sum of:
        #   - ('NodeDepth', 'XShift', 'YShift') * number of tiltmeters
        #   - any of
        #       - ('NodeDepth', 'water_level') * number of piezometers
        #       - ('pressure',) * number of barometers
        #       - ('precipitation',) * number of pluviometers
        # print(f'start: {x.shape}')
        # transpose input from (batch, days, flattened features)
        #   to (batch, flattened features, days)
        x = torch.transpose(x, self.features_position, self.num_day_position)
        # print(f'switch days and features: {x.shape}')
        # counter = 0
        for layer in self.convolutional_layers:
            x = layer(x)
            # print(f'{counter}° convolution: {x.shape}')
            if self.pooling_operation is not None:
                x = self.generic_pooling(x)
                # print(f'{counter}° pooling: {x.shape}')
            # counter += 1

        # reshape input from (batch, flattened features, convolutional output)
        #   to (batch, flattened features * convolutional output)
        x = self.flatten_input(x)
        # print(f'flatten: {x.shape}')

        # counter = 0
        for layer in self.dense_layers:
            x = layer(x)
            # print(f'{counter}° dense layer: {x.shape}')
            x = self.generic_relu(x)
            # print(f'{counter}° relu layer: {x.shape}')
            # counter += 1

        # Add back the time dimension
        # Shape: (outputs) => (num_days, outputs)
        x = torch.reshape(x, (-1, *self.num_output))
        # print(f'reshape: {x.shape}')
        return x

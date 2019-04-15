
import torch

from .nac import NACLayer, NACCell
from .nalu import NALULayer, NALUCell
from .gumbel_nac import GumbelNACLayer, GumbelNACCell
from .gumbel_nalu import GumbelNALULayer, GumbelNALUCell
from .regualized_linear_nac import RegualizedLinearNACLayer, RegualizedLinearNACCell
from .regualized_linear_nalu import RegualizedLinearNALULayer, RegualizedLinearNALUCell
from .basic import BasicLayer, BasicCell
from ..abstract import ExtendedTorchModule

unit_name_to_layer_class = {
    'NAC': NACLayer,
    'NALU': NALULayer,
    'GumbelNAC': NACLayer,
    'GumbelNALU': GumbelNALULayer,
    'RegualizedLinearNAC': RegualizedLinearNACLayer,
    'RegualizedLinearNALU': RegualizedLinearNACLayer,
}

unit_name_to_cell_class = {
    'NAC': NACCell,
    'NALU': NALUCell,
    'GumbelNAC': GumbelNACCell,
    'GumbelNALU': GumbelNALUCell,
    'RegualizedLinearNAC': RegualizedLinearNACCell,
    'RegualizedLinearNALU': RegualizedLinearNACCell,
}

class GeneralizedLayer(ExtendedTorchModule):
    """Abstracts all layers, both basic, NAC and NALU

    Arguments:
        in_features: number of ingoing features
        out_features: number of outgoing features
        unit_name: name of the unit (e.g. NAC, Sigmoid, Tanh)
    """
    UNIT_NAMES = set(unit_name_to_layer_class.keys()) | BasicLayer.ACTIVATIONS

    def __init__(self, in_features, out_features, unit_name, writer=None, **kwags):
        super().__init__('layer', writer=writer, **kwags)
        self.in_features = in_features
        self.out_features = out_features
        self.unit_name = unit_name

        if unit_name in unit_name_to_layer_class:
            Layer = unit_name_to_layer_class[unit_name]
            self.layer = Layer(in_features, out_features,
                               writer=self.writer,
                               **kwags)
        else:
            self.layer = BasicLayer(in_features, out_features,
                                    activation=unit_name,
                                    writer=self.writer,
                                    **kwags)

    def reset_parameters(self):
        self.layer.reset_parameters()

    def forward(self, input):
        return self.layer(input)

    def extra_repr(self):
        return 'in_features={}, out_features={}, unit_name={}'.format(
            self.in_features, self.out_features, self.unit_name
        )

class GeneralizedCell(ExtendedTorchModule):
    """Abstracts all cell, RNN-tanh, RNN-ReLU, GRU, LSTM, NAC and NALU

    Arguments:
        input_size: number of ingoing features
        hidden_size: number of outgoing features
        unit_name: name of the unit (e.g. RNN-tanh, LSTM, NAC)
    """
    UNIT_NAMES = set(unit_name_to_cell_class.keys()) | {'GRU', 'LSTM', 'RNN-tanh', 'RNN-ReLU', 'RNN-linear'}

    def __init__(self, input_size, hidden_size, unit_name, writer=None, **kwags):
        super().__init__('cell', writer=writer, **kwags)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.unit_name = unit_name

        if unit_name in unit_name_to_cell_class:
            Cell = unit_name_to_cell_class[unit_name]
            self.cell = Cell(input_size, hidden_size,
                             writer=self.writer,
                             **kwags)
        elif unit_name == 'none':
            self.cell = PassThoughCell(input_size, hidden_size,
                                       **kwags)
        elif unit_name == 'GRU':
            self.cell = torch.nn.GRUCell(input_size, hidden_size,
                                          **kwags)
        elif unit_name == 'LSTM':
            self.cell = torch.nn.LSTMCell(input_size, hidden_size,
                                          **kwags)
        elif unit_name == 'RNN-tanh':
            self.cell = torch.nn.RNNCell(input_size, hidden_size,
                                         nonlinearity='tanh',
                                         **kwags)
        elif unit_name == 'RNN-ReLU':
            self.cell = torch.nn.RNNCell(input_size, hidden_size,
                                         nonlinearity='relu',
                                         **kwags)
        elif unit_name == 'RNN-linear':
            self.cell = BasicCell(input_size, hidden_size,
                                  activation='linear',
                                  writer=self.writer,
                                  **kwags)
        else:
            raise NotImplementedError(
                f'{unit_name} is not an implemented cell type')

    def reset_parameters(self):
        self.cell.reset_parameters()

    def forward(self, x_t, h_tm1):
        return self.cell(x_t, h_tm1)

    def extra_repr(self):
        return 'input_size={}, hidden_size={}, unit_name={}'.format(
            self.input_size, self.hidden_size, self.unit_name
        )

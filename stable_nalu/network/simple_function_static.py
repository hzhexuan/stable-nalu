
import torch
from ..abstract import ExtendedTorchModule
from ..layer import GeneralizedLayer, BasicLayer

class SimpleFunctionStaticNetwork(ExtendedTorchModule):
    UNIT_NAMES = GeneralizedLayer.UNIT_NAMES

    def __init__(self, unit_name, input_size=100, hidden_size=2, writer=None, first_layer=None, nac_mul='none', eps=1e-7, **kwags):
        super().__init__('network', writer=writer, **kwags)
        self.unit_name = unit_name
        self.input_size = input_size
        self.nac_mul = nac_mul
        self.eps = eps

        if first_layer is not None:
            unit_name_1 = first_layer
        else:
            unit_name_1 = unit_name

        self.layer_1 = GeneralizedLayer(input_size, hidden_size,
                                        unit_name_1,
                                        writer=self.writer,
                                        name='layer_1',
                                        eps=eps, **kwags)

        if nac_mul == 'mnac':
            unit_name_2 = unit_name[0:-3] + 'MNAC'
        else:
            unit_name_2 = unit_name

        self.layer_2 = GeneralizedLayer(hidden_size, 1,
                                        'linear' if unit_name_2 in BasicLayer.ACTIVATIONS else unit_name_2,
                                        writer=self.writer,
                                        name='layer_2',
                                        eps=eps, **kwags)
        self.reset_parameters()
        self.z_1_stored = None

    def reset_parameters(self):
        self.layer_1.reset_parameters()
        self.layer_2.reset_parameters()

    def regualizer(self):
        if self.nac_mul == 'max-safe':
            return super().regualizer({
                'z': torch.mean(torch.relu(1 - self.z_1_stored))
            })
        else:
            return super().regualizer()

    def forward(self, input):
        self.writer.add_summary('x', input)
        z_1 = self.layer_1(input)
        self.z_1_stored = z_1
        self.writer.add_summary('z_1', z_1)

        if self.nac_mul == 'none' or self.nac_mul == 'mnac':
            z_2 = self.layer_2(z_1)
        elif self.nac_mul == 'normal':
            z_2 = torch.exp(self.layer_2(torch.log(torch.abs(z_1) + self.eps)))
        elif self.nac_mul == 'safe':
            z_2 = torch.exp(self.layer_2(torch.log(torch.abs(z_1 - 1) + 1)))
        elif self.nac_mul == 'max-safe':
            z_2 = torch.exp(self.layer_2(torch.log(torch.relu(z_1 - 1) + 1)))
        else:
            raise ValueError(f'Unsupported nac_mul option ({self.nac_mul})')

        self.writer.add_summary('z_2', z_2)
        return z_2

    def extra_repr(self):
        return 'unit_name={}, input_size={}'.format(
            self.unit_name, self.input_size
        )

    
class ReversedFunctionStaticNetwork(ExtendedTorchModule):
    UNIT_NAMES = GeneralizedLayer.UNIT_NAMES

    def __init__(self, unit_name, input_size=100, hidden_size=2, writer=None, first_layer=None, nac_mul='none', eps=1e-7, **kwags):
        super().__init__('network', writer=writer, **kwags)
        self.unit_name = unit_name
        self.input_size = input_size
        self.nac_mul = nac_mul
        self.eps = eps
        if nac_mul == 'mnac':
            unit_name_1 = unit_name[0:-3] + 'MNAC'
        else:
            unit_name_1 = unit_name
            
        self.layer_1 = GeneralizedLayer(input_size, hidden_size,
                                        'linear' if unit_name_1 in BasicLayer.ACTIVATIONS else unit_name_1,
                                        writer=self.writer,
                                        name='layer_1',
                                        eps=eps, **kwags)
        if first_layer is not None:
            unit_name_2 = first_layer
        else:
            unit_name_2 = unit_name
        
        self.layer_2 = GeneralizedLayer(hidden_size, 1,
                                        unit_name_2,
                                        writer=self.writer,
                                        name='layer_2',
                                        eps=eps, **kwags)
        print(unit_name_1, unit_name_2)
        self.reset_parameters()
        self.z_1_stored = None

    def reset_parameters(self):
        self.layer_1.reset_parameters()
        self.layer_2.reset_parameters()

    def regualizer(self):
        return super().regualizer()

    def forward(self, input):
        self.writer.add_summary('x', input)

        if self.nac_mul == 'none' or self.nac_mul == 'mnac':
            z_1 = self.layer_1(input)
        elif self.nac_mul == 'normal':
            z_1 = torch.exp(self.layer_1(torch.log(torch.abs(input) + self.eps)))
        elif self.nac_mul == 'safe':
            z_1 = torch.exp(self.layer_1(torch.log(torch.abs(input - 1) + 1)))
        elif self.nac_mul == 'max-safe':
            z_1 = torch.exp(self.layer_1(torch.log(torch.relu(input - 1) + 1)))
        else:
            raise ValueError(f'Unsupported nac_mul option ({self.nac_mul})')
        self.writer.add_summary('z_1', z_1)
        self.z_1_stored = z_1
        
        z_2 = self.layer_2(z_1)
        self.writer.add_summary('z_2', z_2)
        return z_2

    def extra_repr(self):
        return 'unit_name={}, input_size={}'.format(
            self.unit_name, self.input_size
        )

class MultiFunctionStaticNetwork(ExtendedTorchModule):
    UNIT_NAMES = GeneralizedLayer.UNIT_NAMES

    def __init__(self, unit_name, input_size=100, hidden_size = [8, 32], writer=None, first_layer=None, nac_mul='none', eps=1e-7, **kwags):
        super().__init__('network', writer=writer, **kwags)
        self.unit_name = unit_name
        self.input_size = input_size
        self.nac_mul = nac_mul
        self.eps = eps
        hidden_size.insert(0, input_size)
        l = len(hidden_size)
        self.l = l
        hidden_size.append(1)
        
        for i in range(l):
            if(i % 2 == 0):
                setattr(self,'layer'+str(i+1), GeneralizedLayer(hidden_size[i], hidden_size[i+1],
                                        unit_name,
                                        writer=self.writer,
                                        name='layer'+str(i+1),
                                        eps=eps, **kwags))
            else:
                setattr(self,'layer'+str(i+1), GeneralizedLayer(hidden_size[i], hidden_size[i+1],
                                        unit_name[0:-3] + 'MNAC',
                                        writer=self.writer,
                                        name='layer'+str(i+1),
                                        eps=eps, **kwags))
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.l):
            getattr(self,'layer'+str(i+1)).reset_parameters()

    def regualizer(self):
        return super().regualizer()

    def forward(self, input):
        self.writer.add_summary('x', input)
        for i in range(self.l):
            input = getattr(self,'layer'+str(i+1))(input)
            self.writer.add_summary('z_' + str(i+1), input)
        return input

    def extra_repr(self):
        return 'unit_name={}, input_size={}'.format(
            self.unit_name, self.input_size
        )

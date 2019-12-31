import os
parser = argparse.ArgumentParser(description='Runs the simple function static task')
parser.add_argument('--layer-type',
                    action='store',
                    default='NALU',
                    choices=list(stable_nalu.network.SimpleFunctionStaticNetwork.UNIT_NAMES),
                    type=str,
                    help='Specify the layer type, e.g. Tanh, ReLU, NAC, NALU')
parser.add_argument('--operation',
                    action='store',
                    default='add',
                    choices=[
                        'add', 'sub', 'mul', 'div', 'squared', 'root'
                    ],
                    type=str,
                    help='Specify the operation to use, e.g. add, mul, squared')
parser.add_argument('--num-subsets',
                    action='store',
                    default=2,
                    type=int,
                    help='Specify the number of subsets to use')
parser.add_argument('--regualizer',
                    action='store',
                    default=10,
                    type=float,
                    help='Specify the regualization lambda to be used')
parser.add_argument('--regualizer-z',
                    action='store',
                    default=0,
                    type=float,
                    help='Specify the z-regualization lambda to be used')
parser.add_argument('--regualizer-oob',
                    action='store',
                    default=1,
                    type=float,
                    help='Specify the oob-regualization lambda to be used')
parser.add_argument('--first-layer',
                    action='store',
                    default=None,
                    help='Set the first layer to be a different type')

parser.add_argument('--max-iterations',
                    action='store',
                    default=100000,
                    type=int,
                    help='Specify the max number of iterations to use')
parser.add_argument('--batch-size',
                    action='store',
                    default=128,
                    type=int,
                    help='Specify the batch-size to be used for training')
parser.add_argument('--seed',
                    action='store',
                    default=0,
                    type=int,
                    help='Specify the seed to use')

parser.add_argument('--interpolation-range',
                    action='store',
                    default=[1,2],
                    type=ast.literal_eval,
                    help='Specify the interpolation range that is sampled uniformly from')
parser.add_argument('--extrapolation-range',
                    action='store',
                    default=[2,6],
                    type=ast.literal_eval,
                    help='Specify the extrapolation range that is sampled uniformly from')
parser.add_argument('--input-size',
                    action='store',
                    default=100,
                    type=int,
                    help='Specify the input size')
parser.add_argument('--subset-ratio',
                    action='store',
                    default=0.25,
                    type=float,
                    help='Specify the subset-size as a fraction of the input-size')
parser.add_argument('--overlap-ratio',
                    action='store',
                    default=0.5,
                    type=float,
                    help='Specify the overlap-size as a fraction of the input-size')
parser.add_argument('--simple',
                    action='store_true',
                    default=False,
                    help='Use a very simple dataset with t = sum(v[0:2]) + sum(v[4:6])')

parser.add_argument('--hidden-size',
                    nargs='+', 
                    type=int,
                    help='Specify the vector size of the hidden layer.')
parser.add_argument('--nac-mul',
                    action='store',
                    default='none',
                    choices=['none', 'normal', 'safe', 'max-safe', 'mnac'],
                    type=str,
                    help='Make the second NAC a multiplicative NAC, used in case of a just NAC network.')
parser.add_argument('--oob-mode',
                    action='store',
                    default='clip',
                    choices=['regualized', 'clip'],
                    type=str,
                    help='Choose of out-of-bound should be handled by clipping or regualization.')
parser.add_argument('--regualizer-scaling',
                    action='store',
                    default='linear',
                    choices=['exp', 'linear'],
                    type=str,
                    help='Use an expoentational scaling from 0 to 1, or a linear scaling.')
parser.add_argument('--regualizer-scaling-start',
                    action='store',
                    default=1000000,
                    type=int,
                    help='Start linear scaling at this global step.')
parser.add_argument('--regualizer-scaling-end',
                    action='store',
                    default=2000000,
                    type=int,
                    help='Stop linear scaling at this global step.')
parser.add_argument('--regualizer-shape',
                    action='store',
                    default='linear',
                    choices=['squared', 'linear'],
                    type=str,
                    help='Use either a squared or linear shape for the bias and oob regualizer.')
parser.add_argument('--mnac-epsilon',
                    action='store',
                    default=0,
                    type=float,
                    help='Set the idendity epsilon for MNAC.')
parser.add_argument('--nalu-bias',
                    action='store_true',
                    default=False,
                    help='Enables bias in the NALU gate')
parser.add_argument('--nalu-two-nac',
                    action='store_true',
                    default=False,
                    help='Uses two independent NACs in the NALU Layer')
parser.add_argument('--nalu-two-gate',
                    action='store_true',
                    default=False,
                    help='Uses two independent gates in the NALU Layer')
parser.add_argument('--nalu-mul',
                    action='store',
                    default='normal',
                    choices=['normal', 'safe', 'trig', 'max-safe', 'mnac'],
                    help='Multplication unit, can be normal, safe, trig')
parser.add_argument('--nalu-gate',
                    action='store',
                    default='normal',
                    choices=['normal', 'regualized', 'obs-gumbel', 'gumbel'],
                    type=str,
                    help='Can be normal, regualized, obs-gumbel, or gumbel')

parser.add_argument('--optimizer',
                    action='store',
                    default='adam',
                    choices=['adam', 'sgd'],
                    type=str,
                    help='The optimization algorithm to use, Adam or SGD')
parser.add_argument('--learning-rate',
                    action='store',
                    default=1e-3,
                    type=float,
                    help='Specify the learning-rate')
parser.add_argument('--momentum',
                    action='store',
                    default=0.0,
                    type=float,
                    help='Specify the nestrov momentum, only used with SGD')

parser.add_argument('--no-cuda',
                    action='store_true',
                    default=False,
                    help=f'Force no CUDA (cuda usage is detected automatically as {torch.cuda.is_available()})')
parser.add_argument('--name-prefix',
                    action='store',
                    default='simple_function_static',
                    type=str,
                    help='Where the data should be stored')
parser.add_argument('--remove-existing-data',
                    action='store_true',
                    default=False,
                    help='Should old results be removed')
parser.add_argument('--verbose',
                    action='store_true',
                    default=False,
                    help='Should network measures (e.g. gates) and gradients be shown')
parser.add_argument('--size',
                    type=int,
                    default=2)
parser.add_argument('--percent',
                    type=float,
                    default=1)
args = parser.parse_args()
hidden_size = ""
for e in args.hidden_size:
    hidden_size += " "+str(e)
out = os.popen("python experiments/myexp_sparse.py \
    --batch-size 1024 --learning-rate 0.01 --regualizer 10 \
    --regualizer-scaling-start 1000 --regualizer-scaling-end 2000 \
    --input-size 4 --interpolation-range [-1,1] --extrapolation-range [-1,1] \
    --size args.size --hidden-size" + hidden_size + " --momentum args.momentum --percent args.percent \
    --operation mul --layer-type ReRegualizedLinearNAC --nac-mul mnac \
    --seed 1 --max-iterations 2000 --verbose \
    --name-prefix test --remove-existing-data")

print(out.read())

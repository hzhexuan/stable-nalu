import numpy as np
import os
import ast
import math
import torch
import stable_nalu
import argparse

# Parse arguments
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
                    type=int,
                    help='Specify the vector size of the hidden layer.')
parser.add_argument('--output-c',
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
                    help='Force no CUDA (cuda usage is detected automatically as {torch.cuda.is_available()})')
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

if(args.percent == 1):
  mask = np.ones((args.size, args.size))
else:
  mask = np.random.binomial(1, args.percent, (args.size, args.size))
 
def log(x):
  return torch.sign(x) * torch.log(torch.abs(x)+1e-16)

def Dataset(num, extra=False):
  x = 2 * np.random.rand(num, args.size, args.size) - 1
  if(extra==True):
    x = x + np.sign(x)
  x = x * mask
  t = np.linalg.det(x).reshape([-1, 1])
  x = torch.Tensor(np.transpose(x[...,np.newaxis], [0,3,1,2]))
  if(torch.cuda.is_available()):
    return x.cuda(), torch.Tensor(t).cuda()
  return x, torch.Tensor(t)

setattr(args, 'cuda', torch.cuda.is_available() and not args.no_cuda)
# Prepear logging
summary_writer = stable_nalu.writer.SummaryWriter(
    f'{args.name_prefix}/{args.layer_type.lower()}'
    f'{"-nac-" if args.nac_mul != "none" else ""}'
    f'{"n" if args.nac_mul == "normal" else ""}'
    f'{"s" if args.nac_mul == "safe" else ""}'
    f'{"s" if args.nac_mul == "max-safe" else ""}'
    f'{"t" if args.nac_mul == "trig" else ""}'
    f'{"m" if args.nac_mul == "mnac" else ""}'
    f'{"-nalu-" if (args.nalu_bias or args.nalu_two_nac or args.nalu_two_gate or args.nalu_mul != "normal" or args.nalu_gate != "normal") else ""}'
    f'{"b" if args.nalu_bias else ""}'
    f'{"2n" if args.nalu_two_nac else ""}'
    f'{"2g" if args.nalu_two_gate else ""}'
    f'{"s" if args.nalu_mul == "safe" else ""}'
    f'{"s" if args.nalu_mul == "max-safe" else ""}'
    f'{"t" if args.nalu_mul == "trig" else ""}'
    f'{"m" if args.nalu_mul == "mnac" else ""}'
    f'{"r" if args.nalu_gate == "regualized" else ""}'
    f'{"u" if args.nalu_gate == "gumbel" else ""}'
    f'{"uu" if args.nalu_gate == "obs-gumbel" else ""}'
    f'_op-{args.operation.lower()}'
    f'_oob-{"c" if args.oob_mode == "clip" else "r"}'
    f'_rs-{args.regualizer_scaling}-{args.regualizer_shape}'
    f'_eps-{args.mnac_epsilon}'
    f'_rl-{args.regualizer_scaling_start}-{args.regualizer_scaling_end}'
    f'_r-{args.regualizer}-{args.regualizer_z}-{args.regualizer_oob}'
    f'_i-{args.interpolation_range[0]}-{args.interpolation_range[1]}'
    f'_e-{args.extrapolation_range[0]}-{args.extrapolation_range[1]}'
    f'_z-{"simple" if args.simple else f"{args.input_size}-{args.subset_ratio}-{args.overlap_ratio}"}'
    f'_b{args.batch_size}'
    f'_s{args.seed}'
    f'_z{args.num_subsets}'
    f'_lr-{args.optimizer}-{"%.5f" % args.learning_rate}-{args.momentum}',
    remove_existing_data=args.remove_existing_data
)

# Set threads
if 'LSB_DJOB_NUMPROC' in os.environ:
    torch.set_num_threads(int(os.environ['LSB_DJOB_NUMPROC']))

# Set seed
#torch.manual_seed(args.seed)
#torch.backends.cudnn.deterministic = True

# setup model
model = stable_nalu.network.ConvStaticNetwork(
    input_c=1, output_c=args.output_c, kernel=3, hidden_size=args.hidden_size, input_size=args.size,
    writer=summary_writer.every(1000).verbose(args.verbose),
    nac_oob=args.oob_mode,
    regualizer_shape=args.regualizer_shape,
    regualizer_z=args.regualizer_z,
    mnac_epsilon=args.mnac_epsilon,
    nac_mul=args.nac_mul,
    nalu_bias=args.nalu_bias,
    nalu_two_nac=args.nalu_two_nac,
    nalu_two_gate=args.nalu_two_gate,
    nalu_mul=args.nalu_mul,
    nalu_gate=args.nalu_gate,
)

model.reset_parameters()
if args.cuda:
    model.cuda()
criterion = torch.nn.MSELoss()

for module in model.children():
  print(module)

if args.optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
elif args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
else:
    raise ValueError(f'{args.optimizer} is not a valid optimizer algorithm')

def test_model(data):
    with torch.no_grad(), model.no_internal_logging(), model.no_random():
        x, t = data
        return criterion(model(x), t)

dataset_valid_interpolation_data = Dataset(256)
dataset_test_extrapolation_data = Dataset(256, extra=True)

# Test with different initialization
L_in = []
L_ex = []
for i in range(100):
  model.reset_parameters()
  L_in.append(test_model(dataset_valid_interpolation_data).cpu().numpy())
  L_ex.append(test_model(dataset_test_extrapolation_data).cpu().numpy())
#print("in, ex", sum(L_in)/len(L_in), sum(L_ex)/len(L_ex))
  
# Train model
print('')
for epoch_i in range(args.max_iterations + 1):
    x_train, t_train = Dataset(args.batch_size)
    summary_writer.set_iteration(epoch_i)

    # Prepear model
    model.set_parameter('tau', max(0.5, math.exp(-1e-5 * epoch_i)))
    optimizer.zero_grad()

    # Log validation
    if epoch_i % 1000 == 0:
        interpolation_error = test_model(dataset_valid_interpolation_data)
        extrapolation_error = test_model(dataset_test_extrapolation_data)

        summary_writer.add_scalar('metric/valid/interpolation', interpolation_error)
        summary_writer.add_scalar('metric/test/extrapolation', extrapolation_error)

    # forward
    y_train = model(x_train)
    regualizers = model.regualizer()

    if (args.regualizer_scaling == 'linear'):
        r_w_scale = max(0, min(1, (
            (epoch_i - args.regualizer_scaling_start) /
            (args.regualizer_scaling_end - args.regualizer_scaling_start)
        )))
    elif (args.regualizer_scaling == 'exp'):
        r_w_scale = 1 - math.exp(-1e-5 * epoch_i)

    loss_train_criterion = criterion(y_train, t_train)
    loss_train_regualizer = args.regualizer * r_w_scale * regualizers['W'] + regualizers['g'] + args.regualizer_z * regualizers['z'] + args.regualizer_oob * regualizers['W-OOB']
    loss_train = loss_train_criterion + loss_train_regualizer

    # Log loss
    if args.verbose or epoch_i % 1000 == 0:
        summary_writer.add_scalar('loss/train/critation', loss_train_criterion)
        summary_writer.add_scalar('loss/train/regualizer', loss_train_regualizer)
        summary_writer.add_scalar('loss/train/total', loss_train)
    if epoch_i % 1000 == 0:
        print('train %d: %.5f, inter: %.5f, extra: %.5f' % (epoch_i, loss_train_criterion, interpolation_error, extrapolation_error))

    # Optimize model
    if loss_train.requires_grad:
        loss_train.backward()
        optimizer.step()
    model.optimize(loss_train_criterion)

    # Log gradients if in verbose mode
    if args.verbose and epoch_i % 1000 == 0:
        model.log_gradients()
    if(epoch_i%1000 == 0):
      x, t= dataset_test_extrapolation_data
      print(model(x)[0], t[0])
# Compute validation loss
loss_valid_inter = test_model(dataset_valid_interpolation_data)
loss_valid_extra = test_model(dataset_test_extrapolation_data)

# Write results for this training
#print(f'finished:')
#print(f'  - loss_train: {loss_train}')
#print(f'  - loss_valid_inter: {loss_valid_inter}')
#print(f'  - loss_valid_extra: {loss_valid_extra}')
def listToString(s):   
    return ' '.join([str(elem) for elem in s])

print(sum(L_in)/len(L_in), sum(L_ex)/len(L_ex), loss_valid_inter.cpu().numpy(), loss_valid_extra.cpu().numpy())
with open("result"+str(args.size)+".txt", "a+") as f:
  f.write(listToString([sum(L_in)/len(L_in), sum(L_ex)/len(L_ex), loss_valid_inter.cpu().numpy(), loss_valid_extra.cpu().numpy()]) + '\n')
# Use saved weights to visualize the intermediate values.
#stable_nalu.writer.save_model(summary_writer.name, model)
import sys
#sys.exit([sum(L_in)/len(L_in), sum(L_ex)/len(L_ex), loss_valid_inter.cpu().numpy(), loss_valid_extra.cpu().numpy()])
os._exit(0)

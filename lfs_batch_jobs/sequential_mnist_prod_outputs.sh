#!/bin/bash
export LSB_JOB_REPORT_MAIL=N
experiment_name='sequential_mnist_prod_outputs'
mnist_outputs=( 1 2 3 4 5 6 )

for seed in {0..4}
do
    for mnist_output_size in "${mnist_outputs[@]}"
    do
        bsub -q gpuv100 -n 2 -gpu "num=1:mode=exclusive_process" -W 6:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=2GB]" ./python_lfs_job.sh \
            experiments/sequential_mnist.py \
            --operation cumprod --layer-type NAC --nac-mul normal \
            --mnist-digits 123456789 --mnist-outputs ${mnist_output_size} \
            --interpolation-length 2 --extrapolation-lengths '[1,2,3,4,5,6,7,8,9]' \
            --seed ${seed} --max-epochs 1000 --verbose \
            --name-prefix ${experiment_name} --remove-existing-data

        bsub -q gpuv100 -n 2 -gpu "num=1:mode=exclusive_process" -W 6:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=2GB]" ./python_lfs_job.sh \
            experiments/sequential_mnist.py \
            --operation cumprod --layer-type PosNAC --nac-mul normal \
            --mnist-digits 123456789 --mnist-outputs ${mnist_output_size} \
            --interpolation-length 2 --extrapolation-lengths '[1,2,3,4,5,6,7,8,9]' \
            --seed ${seed} --max-epochs 1000 --verbose \
            --name-prefix ${experiment_name} --remove-existing-data

        bsub -q gpuv100 -n 2 -gpu "num=1:mode=exclusive_process" -W 6:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=2GB]" ./python_lfs_job.sh \
            experiments/sequential_mnist.py \
            --operation cumprod --layer-type NALU \
            --mnist-digits 123456789 --mnist-outputs ${mnist_output_size} \
            --interpolation-length 2 --extrapolation-lengths '[1,2,3,4,5,6,7,8,9]' \
            --seed ${seed} --max-epochs 1000 --verbose \
            --name-prefix ${experiment_name} --remove-existing-data

        bsub -q gpuv100 -n 2 -gpu "num=1:mode=exclusive_process" -W 6:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=2GB]" ./python_lfs_job.sh \
            experiments/sequential_mnist.py \
            --operation cumprod --layer-type ReRegualizedLinearNAC --nac-mul mnac \
            --mnist-digits 123456789 --mnist-outputs ${mnist_output_size} \
            --regualizer-z 1 \
            --interpolation-length 2 --extrapolation-lengths '[1,2,3,4,5,6,7,8,9]' \
            --seed ${seed} --max-epochs 1000 --verbose \
            --name-prefix ${experiment_name} --remove-existing-data

        bsub -q gpuv100 -n 2 -gpu "num=1:mode=exclusive_process" -W 6:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=2GB]" ./python_lfs_job.sh \
            experiments/sequential_mnist.py \
            --operation cumprod --layer-type ReRegualizedLinearPosNAC --nac-mul normal \
            --mnist-digits 123456789 --mnist-outputs ${mnist_output_size} \
            --interpolation-length 2 --extrapolation-lengths '[1,2,3,4,5,6,7,8,9]' \
            --regualizer-z 1 \
            --seed ${seed} --max-epochs 1000 --verbose \
            --name-prefix ${experiment_name} --remove-existing-data

        bsub -q gpuv100 -n 2 -gpu "num=1:mode=exclusive_process" -W 6:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=2GB]" ./python_lfs_job.sh \
            experiments/sequential_mnist.py \
            --operation cumprod --layer-type LSTM \
            --mnist-digits 123456789 --mnist-outputs ${mnist_output_size} \
            --interpolation-length 2 --extrapolation-lengths '[1,2,3,4,5,6,7,8,9]' \
            --seed ${seed} --max-epochs 1000 --verbose \
            --name-prefix ${experiment_name} --remove-existing-data
    done
done

import os
out = os.popen("python experiments/myexp_sparse.py --batch-size 1024 --learning-rate 0.01 --regualizer 20 --regualizer-scaling-start 100000 --regualizer-scaling-end 200000 \
    --input-size 4 --interpolation-range [-1,1] --extrapolation-range [-1,1] \
    --size 6 --hidden-size 16 24  --momentum 0.0 --percent 0.4\
    --operation mul --layer-type ReRegualizedLinearNAC --nac-mul mnac \
    --seed 1 --max-iterations 200000 --verbose \
    --name-prefix test --remove-existing-data")

print(out.read())

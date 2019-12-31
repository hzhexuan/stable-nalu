
for i in $(seq 1 1 $1);do
    {python experiments/myexp_sparse.py --batch-size 64 --learning-rate 0.01 --regualizer 20 --regualizer-scaling-start 10000 --regualizer-scaling-end 20000 \
        --input-size 4 --interpolation-range [-1,1] --extrapolation-range [-1,1] \
        --size $2 --hidden-size 16 24  --momentum 0.0 --percent $3\
        --operation mul --layer-type ReRegualizedLinearNAC --nac-mul mnac \
        --seed $i --max-iterations 20000 \
        --name-prefix test --remove-existing-data}&
done
wait

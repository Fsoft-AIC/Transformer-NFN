NUM_PROCESSS_PER_GPU=4
GPUS=0,1,2,3,4,5
SWEEP_ID=default

# run NUM_PROCESSS_PER_GPU processes on each GPU
for i in $(seq 0 $((NUM_PROCESSS_PER_GPU-1))); do
    for j in $(echo $GPUS | sed "s/,/ /g"); do
        CUDA_VISIBLE_DEVICES=$j wandb agent $SWEEP_ID &
    done
done

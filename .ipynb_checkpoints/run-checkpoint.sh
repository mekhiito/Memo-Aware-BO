# kill -9 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
exp_group=$(date '+%Y%m%d-%H%M')
ACQF_ARRAY=("EEIPU")

for acqf in ${ACQF_ARRAY[@]}; do
    for trial in {1..10}; do
        CUDA_VISIBLE_DEVICES=0 python main.py --trial-num $trial --exp-group $exp_group --acqf $acqf &
    done;
done;

# ACQF_ARRAY=("EI" "EIPS")

# for acqf in ${ACQF_ARRAY[@]}; do
#     for trial in {1..10}; do
#         CUDA_VISIBLE_DEVICES=1 python main.py --trial-num $trial --exp-group $exp_group --acqf $acqf &
#     done;
# done;

wait
set -ex
# models
RESULTS_DIR='./results/base_gray_color'

# dataset
CLASS='base_gray_color'

DIRECTION='AtoC' # 'AtoB' or 'BtoC'
LOAD_SIZE=64
FINE_SIZE=64
RESIZE_OR_CROP='none'
NO_FLIP='--no_flip'
NITER=30
NITER_DECAY=50
SAVE_EPOCH=10
NEF=64
NGF=32
NDF=32
NET_G='dualnet'
NET_D='basic_64_multi'
NET_D2='basic_64_multi'
NET_E='resnet_64'
LAMBDA_L1=20.0
DATASET_MODE='multi_fusion'
USE_ATTENTION='--use_attention'
WHERE_ADD='all'
CONDITIONAL_D='--conditional_D'

# misc
GPU_ID=0   # gpu id


# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python ./test.py \
  --dataroot ./datasets/${CLASS} \
  --results_dir ${RESULTS_DIR} \
  --checkpoints_dir ./pretrained_models/ \
  --name ${CLASS} \
  --direction ${DIRECTION} \
  --loadSize ${FINE_SIZE} \
  --fineSize ${FINE_SIZE} \
  --input_nc ${INPUT_NC} \
  --aspect_ratio ${ASPECT_RATIO} \
  --no_flip
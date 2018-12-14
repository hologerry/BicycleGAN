set -ex
# models
RESULTS_DIR='./results/skeleton_gray_color'
MODEL='dualnet'


# dataset
CLASS='skeleton_gray_color'

PHASE='test'

DIRECTION='AtoC' # 'AtoB' or 'BtoC'
LOAD_SIZE=64
FINE_SIZE=64
INPUT_NC=3
NENCODE=10
FEW_SIZE=10
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
DATASET_MODE='cn_multi_fusion'
USE_ATTENTION='--use_attention'
WHERE_ADD='all'
CONDITIONAL_D='--conditional_D'

NUM_TEST=2000

# misc
GPU_ID=$1   # gpu id


# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python ./test.py \
  --dataroot ./datasets/${CLASS} \
  --results_dir ${RESULTS_DIR} \
  --checkpoints_dir ./pretrained_models/ \
  --nencode ${NENCODE} \
  --few_size ${FEW_SIZE} \
  --name ${CLASS} \
  --phase ${PHASE} \
  --direction ${DIRECTION} \
  --loadSize ${LOAD_SIZE} \
  --fineSize ${FINE_SIZE} \
  --resize_or_crop ${RESIZE_OR_CROP} \
  --input_nc ${INPUT_NC} \
  --model ${MODEL} \
  --ngf ${NGF} \
  --ndf ${NDF} \
  --nef ${NEF} \
  --netG ${NET_G} \
  --netE ${NET_E} \
  --netD ${NET_D} \
  --netD2 ${NET_D2} \
  --dataset_mode ${DATASET_MODE} \
  --num_test ${NUM_TEST} \
  --no_flip
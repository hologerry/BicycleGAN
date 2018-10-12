set -ex
CLASS='edges2shoes'  # facades, day2night, edges2shoes, edges2handbags, maps
MODEL='bicycle_gan'
CLASS=${1}
GPU_ID=${2}

# DISPLAY_ID=$((GPU_ID*10+1))
DISPLAY_ID=0
PORT=2005

NZ=8


CHECKPOINTS_DIR=checkpoints/${CLASS}/  # execute .sh in project root dir to ensure right path
DATE=`date '+%d_%m_%Y_%H'`
NAME=${CLASS}_${MODEL}_${DATE}  # experiment name defined in base_options.py


# dataset
NO_FLIP=''
DIRECTION='AtoB'
LOAD_SIZE=286
FINE_SIZE=256
RESIZE_OR_CROP='resize_and_crop'
INPUT_NC=3

# dataset parameters
case ${CLASS} in
'facades')
  NITER=200
  NITER_DECAY=200
  SAVE_EPOCH=25
  DIRECTION='BtoA'
  ;;
'edges2shoes')
  NITER=30
  NITER_DECAY=30
  LOAD_SIZE=256
  SAVE_EPOCH=5
  INPUT_NC=1
  NO_FLIP='--no_flip'
  ;;
'edges2handbags')
  NITER=15
  NITER_DECAY=15
  LOAD_SIZE=256
  SAVE_EPOCH=5
  INPUT_NC=1
  ;;
'maps')
  NITER=200
  NITER_DECAY=200
  LOAD_SIZE=600
  SAVE_EPOCH=25
  DIRECTION='BtoA'
  ;;
'night2day')
  NITER=50
  NITER_DECAY=50
  SAVE_EPOCH=10
  ;;
'capitals64')
  LOAD_SIZE=64
  FINE_SIZE=64
  RESIZE_OR_CROP='none'
  NITER=50
  NITER_DECAY=50
  SAVE_EPOCH=10
  ;;
'gray2grad0')
  LOAD_SIZE=64
  FINE_SIZE=64
  RESIZE_OR_CROP='none'
  NITER=50
  NITER_DECAY=50
  SAVE_EPOCH=10
  ;;
'gray2grad1')
  LOAD_SIZE=64
  FINE_SIZE=64
  RESIZE_OR_CROP='none'
  NITER=50
  NITER_DECAY=50
  SAVE_EPOCH=10
  ;;
'gray2grad2')
  LOAD_SIZE=64
  FINE_SIZE=64
  RESIZE_OR_CROP='none'
  NITER=50
  NITER_DECAY=50
  SAVE_EPOCH=10
  ;;
*)
  echo 'WRONG category: '${CLASS}
  exit
  ;;
esac



# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python ./train.py \
  --display_id ${DISPLAY_ID} \
  --dataroot ./datasets/${CLASS} \
  --name ${NAME} \
  --model ${MODEL} \
  --display_port ${PORT} \
  --direction ${DIRECTION} \
  --checkpoints_dir ${CHECKPOINTS_DIR} \
  --loadSize ${LOAD_SIZE} \
  --fineSize ${FINE_SIZE} \
  --resize_or_crop ${RESIZE_OR_CROP} \
  --nz ${NZ} \
  --save_epoch_freq ${SAVE_EPOCH} \
  --input_nc ${INPUT_NC} \
  --niter ${NITER} \
  --niter_decay ${NITER_DECAY} \
  --use_dropout

set -ex
# CLASS='edges2shoes'  # facades, day2night, edges2shoes, edges2handbags, maps
MODEL='Dualnet'
CLASS=${1}
GPU_ID=${2}

DISPLAY_ID=$((GPU_ID*10+1))
# DISPLAY_ID=0
PORT=8097

NZ=16


CHECKPOINTS_DIR=checkpoints/${CLASS}/  # execute .sh in project root dir to ensure right path
DATE=`date '+%d_%m_%Y-%H_%M'`
NAME=${CLASS}_${MODEL}_${DATE}  # experiment name defined in base_options.py


# dataset
NO_FLIP=''
DIRECTION='AtoB'
LOAD_SIZE=286
FINE_SIZE=256
RESIZE_OR_CROP='resize_and_crop'
INPUT_NC=3
BATCH_SIZE=16
DATASET_MODE='aligned'
WHERE_ADD='all'
CONDITIONAL_D=''

# Networks module
NGF=64
NDF=64
NEF=64
NET_G='unet_64'
NET_D='basic_64_multi'
NET_E='resnet_64'
USE_ATTENTION=''
USE_SPECTRAL_NORM_G=''
USE_SPECTRAL_NORM_D=''
LAMBDA_L1=10.0

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
'capitals64' | 'gray2grad0' | 'gray2grad1' | 'gray2grad2' | 'mcgan')
  # Deprecated !!!
  DIRECTION='AtoB'
  BATCH_SIZE=16
  LOAD_SIZE=64
  FINE_SIZE=64
  RESIZE_OR_CROP='none'
  NO_FLIP='--no_flip'
  NITER=60
  NITER_DECAY=60
  SAVE_EPOCH=10
  NEF=64
  NGF=32
  NDF=32
  NET_G='dualnet'
  NET_D='basic_64_multi'
  NET_D2='basic_64_multi'
  NET_E='resnet_64'
  LAMBDA_L1=50.0
  DATASET_MODE='multi_aligned'
  # USE_ATTENTION='--use_attention'
  # USE_SPECTRAL_NORM_G='--use_spectral_norm_G'
  # USE_SPECTRAL_NORM_D='--use_spectral_norm_D'
  ;;
'base_gray_color')
  DIRECTION='AtoB' # 'AtoB' or 'BtoC'
  BATCH_SIZE=64
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
  DATASET_MODE='multi_aligned'
  USE_ATTENTION='--use_attention'
  WHERE_ADD='input'
  CONDITIONAL_D='--conditional_D'

  ;;
*)
  echo 'WRONG category: '${CLASS}
  exit
  ;;
esac



# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python3 ./train.py \
  --display_id ${DISPLAY_ID} \
  --dataroot ./datasets/${CLASS} \
  --name ${NAME} \
  --model ${MODEL} \
  --display_port ${PORT} \
  --direction ${DIRECTION} \
  --checkpoints_dir ${CHECKPOINTS_DIR} \
  --batch_size ${BATCH_SIZE} \
  --loadSize ${LOAD_SIZE} \
  --fineSize ${FINE_SIZE} \
  --resize_or_crop ${RESIZE_OR_CROP} \
  ${NO_FLIP} \
  ${USE_ATTENTION} \
  ${USE_SPECTRAL_NORM_G} \
  ${USE_SPECTRAL_NORM_D} \
  --nz ${NZ} \
  --save_epoch_freq ${SAVE_EPOCH} \
  --input_nc ${INPUT_NC} \
  --niter ${NITER} \
  --niter_decay ${NITER_DECAY} \
  --ngf ${NGF} \
  --ndf ${NDF} \
  --nef ${NEF} \
  --netG ${NET_G} \
  --netE ${NET_E} \
  --netD ${NET_D} \
  --netD2 ${NET_D2} \
  --use_dropout \
  --dataset_mode ${DATASET_MODE} \
  --lambda_L1 ${LAMBDA_L1} \
  --where_add ${WHERE_ADD} \
  --conditional_D ${CONDITIONAL_D}

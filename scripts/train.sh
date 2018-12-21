set -ex
# CLASS='edges2shoes'  # facades, day2night, edges2shoes, edges2handbags, maps
MODEL='dualnet'
CLASS=${1}
GPU_ID=${2}

DISPLAY_ID=`date '+%H%M'`
# DISPLAY_ID=0

PORT=9097

NZ=16
NENCODE=4
FEW_SIZE=10

CHECKPOINTS_DIR=checkpoints/${CLASS}/  # execute .sh in project root dir to ensure right path


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

BLACK_EPOCH=0
DISPLAY_FREQ=500

MODEL='bicycle_gan'

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
'base_gray_color')
  MODEL='dualnet'
  DIRECTION='AtoC' # 'AtoB' or 'BtoC'
  NENCODE=4
  BATCH_SIZE=100
  LOAD_SIZE=64
  FINE_SIZE=64
  RESIZE_OR_CROP='none'
  NO_FLIP='--no_flip'
  NITER=50
  NITER_DECAY=100
  SAVE_EPOCH=10
  NEF=64
  NGF=32
  NDF=32
  NET_G='dualnet'
  NET_D='basic_64'
  NET_D2='basic_64'
  NET_R='basic_64'
  NET_E='resnet_64'
  LAMBDA_L1=100.0
  LAMBDA_L1_B=50.0
  LAMBDA_CX=25.0
  LAMBDA_CX_B=15.0
  LAMBDA_L2=100.0
  DATASET_MODE='multi_fusion'
  USE_ATTENTION='--use_attention'
  WHERE_ADD='all'
  CONDITIONAL_D='--conditional_D'
  CONTINUE_TRAIN=''
  BLACK_EPOCH=0
  ;;
'base_gray_texture')
  DATA_ID=${3}     # 0-34 means train the id dataset, 35 means train all the 35 dataset
  CLASS=$CLASS'_'$DATA_ID
  MODEL='dualnet'
  DIRECTION='AtoC' # 'AtoB' or 'BtoC'
  NENCODE=4
  BATCH_SIZE=100
  LOAD_SIZE=64
  FINE_SIZE=64
  RESIZE_OR_CROP='none'
  NO_FLIP='--no_flip'
  NITER=1000
  NITER_DECAY=4000
  SAVE_EPOCH=500
  NEF=64
  NGF=32
  NDF=32
  NET_G='dualnet'
  NET_D='basic_64'
  NET_D2='basic_64'
  NET_R='basic_64'
  NET_E='resnet_64'
  LAMBDA_L1=100.0
  LAMBDA_L1_B=20.0
  LAMBDA_CX=45.0
  LAMBDA_CX_B=35.0
  LAMBDA_L2=100.0
  DATASET_MODE='few_fusion'
  USE_ATTENTION='--use_attention'
  WHERE_ADD='all'
  CONDITIONAL_D='--conditional_D'
  CONTINUE_TRAIN='--continue_train'
  BLACK_EPOCH=0
  DISPLAY_FREQ=100
  ;;
'skeleton_gray_color')
  MODEL='dualnet'
  DIRECTION='AtoC' # 'AtoB' or 'BtoC'
  NENCODE=10
  FEW_SIZE=30
  BATCH_SIZE=100
  LOAD_SIZE=64
  FINE_SIZE=64
  RESIZE_OR_CROP='none'
  NO_FLIP='--no_flip'
  NITER=10
  NITER_DECAY=10
  SAVE_EPOCH=2
  NEF=64
  NGF=32
  NDF=32
  NET_G='dualnet'
  NET_D='basic_64'
  NET_D2='basic_64'
  NET_R='basic_64'
  NET_E='resnet_64'
  LAMBDA_L1=100.0
  LAMBDA_L1_B=50.0
  LAMBDA_CX=25.0
  LAMBDA_CX_B=15.0
  LAMBDA_L2=100.0
  DATASET_MODE='cn_multi_fusion'
  USE_ATTENTION='--use_attention'
  WHERE_ADD='all'
  CONDITIONAL_D='--conditional_D'
  CONTINUE_TRAIN=''
  BLACK_EPOCH=0
  ;;
  'skeleton_gray_texture')
  MODEL='dualnet'
  DIRECTION='AtoC' # 'AtoB' or 'BtoC'
  NENCODE=10
  FEW_SIZE=30
  BATCH_SIZE=100
  LOAD_SIZE=64
  FINE_SIZE=64
  RESIZE_OR_CROP='none'
  NO_FLIP='--no_flip'
  NITER=40
  NITER_DECAY=60
  SAVE_EPOCH=10
  NEF=64
  NGF=32
  NDF=32
  NET_G='dualnet'
  NET_D='basic_64'
  NET_D2='basic_64'
  NET_R='basic_64'
  NET_E='resnet_64'
  LAMBDA_L1=100.0
  LAMBDA_L1_B=20.0
  LAMBDA_CX=25.0
  LAMBDA_CX_B=15.0
  LAMBDA_L2=100.0
  DATASET_MODE='cn_multi_fusion'
  USE_ATTENTION='--use_attention'
  WHERE_ADD='all'
  CONDITIONAL_D='--conditional_D'
  CONTINUE_TRAIN='--continue_train'
  BLACK_EPOCH=0
  DISPLAY_FREQ=100
  ;;
*)
  echo 'WRONG category: '${CLASS}
  exit
  ;;
esac

DATE=`date '+%d_%m_%Y-%H'`      # delete minute for more convinent continue training, just run one experiment in an hour
NAME=${CLASS}_${MODEL}_${DATE}  # experiment name defined in base_options.py

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
  --nencode ${NENCODE} \
  --few_size ${FEW_SIZE} \
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
  --netD_B ${NET_D2} \
  --netR ${NET_R} \
  --use_dropout \
  --dataset_mode ${DATASET_MODE} \
  --lambda_L1 ${LAMBDA_L1} \
  --lambda_L1_B ${LAMBDA_L1_B} \
  --lambda_CX ${LAMBDA_CX} \
  --lambda_CX_B ${LAMBDA_CX_B} \
  --lambda_L2 ${LAMBDA_L2} \
  --where_add ${WHERE_ADD} \
  ${CONDITIONAL_D} \
  ${CONTINUE_TRAIN} \
  --black_epoch_freq ${BLACK_EPOCH} \
  --display_freq ${DISPLAY_FREQ}


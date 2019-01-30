set -ex
# CLASS='edges2shoes'  # facades, day2night, edges2shoes, edges2handbags, maps
MODEL='dualnet'
CLASS=${1}
GPU_ID=${2}


DISPLAY_ID=`date '+%H%M'`
# DISPLAY_ID=0

PORT=10002

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
LAMBDA_L1=10.0

LR=0.0002

BLACK_EPOCH=0
VALIDATE_FREQ=0
DISPLAY_FREQ=100

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
'base_gray_color' | 'base_gray_color_s')
  MODEL='dualnet'
  DIRECTION='AtoC' # 'AtoB' or 'BtoC'
  NENCODE=4
  BATCH_SIZE=80
  LOAD_SIZE=64
  FINE_SIZE=64
  RESIZE_OR_CROP='none'
  NO_FLIP='--no_flip'
  NITER=50
  NITER_DECAY=250
  SAVE_EPOCH=2
  NEF=64
  NGF=32
  NDF=32
  NET_G='dualnet'
  NET_D='basic_64'
  NET_D2='basic_64'
  NET_DLOCAL='basic_32'
  NET_R='basic_64'
  NET_E='resnet_64'
  LAMBDA_L1=100.0
  LAMBDA_L1_B=50.0
  LAMBDA_CX=25.0
  LAMBDA_CX_B=15.0
  LAMBDA_L2=100.0
  LAMBDA_TX=0.0
  LAMBDA_TX_B=0.0
  LAMBDA_PATCH=0
  LAMBDA_LOCAL_D=0
  LAMBDA_LOCAL_STYLE=0
  LAMBDA_GAN=1.0
  LAMBDA_GAN_B=1.0
  DATASET_MODE='multi_fusion'
  USE_ATTENTION='--use_attention'
  WHERE_ADD='all'
  CONDITIONAL_D='--conditional_D'
  CONTINUE_TRAIN=''
  BLACK_EPOCH=0
  PRINT_FREQ=400
  ;;
'base_gray_texture')
  DATA_ID=${3}     # 0-34 means train the id dataset, 35 means train all the 35 dataset
  CLASS=$CLASS'_'$DATA_ID
  MODEL='dualnet'
  DIRECTION='AtoC' # 'AtoB' or 'BtoC'
  NENCODE=4
  BATCH_SIZE=80
  LOAD_SIZE=64
  FINE_SIZE=64
  RESIZE_OR_CROP='none'
  NO_FLIP='--no_flip'
  NITER=500
  NITER_DECAY=2500
  SAVE_EPOCH=100
  NEF=64
  NGF=32
  NDF=32
  NET_G='dualnet'
  NET_D='basic_64'
  NET_D2='basic_64'
  NET_R='basic_64'
  NET_E='resnet_64'
  LAMBDA_L1=100.0
  LAMBDA_L1_B=60.0
  LAMBDA_CX=50.0
  LAMBDA_CX_B=15.0
  LAMBDA_L2=100.0
  DATASET_MODE='few_fusion'
  USE_ATTENTION='--use_attention'
  WHERE_ADD='all'
  CONDITIONAL_D='--conditional_D'
  CONTINUE_TRAIN='--continue_train'
  VALIDATE_FREQ=50
  BLACK_EPOCH=0
  DISPLAY_FREQ=100
  LR=0.00002
  ;;
'base_gray_texture_unpaired' | 'base_gray_texture_unpaired_s')
  MODEL='dualnet'
  DIRECTION='AtoC' # 'AtoB' or 'BtoC'
  NENCODE=4
  BATCH_SIZE=32
  LOAD_SIZE=64
  FINE_SIZE=64
  RESIZE_OR_CROP='none'
  NO_FLIP='--no_flip'
  NITER=500
  NITER_DECAY=2500
  SAVE_EPOCH=10
  NEF=64
  NGF=32
  NDF=32

  NET_G='dualnet'
  NET_D='basic_64'
  NET_D2='basic_64'
  NET_DLOCAL='basic_32'
  NET_E='resnet_64'
  LAMBDA_L1=100.0
  LAMBDA_L1_B=60.0
  LAMBDA_CX=50.0
  LAMBDA_CX_B=15.0
  LAMBDA_L2=100.0
  LAMBDA_TX=0.0
  LAMBDA_TX_B=0.0
  #LAMBDA_L1=100.0
  #LAMBDA_L1_B=60.0
  #LAMBDA_CX=50.0
  #LAMBDA_CX_B=15.0
  #LAMBDA_L2=100.0

  LAMBDA_GAN=1.0
  LAMBDA_GAN_B=1.0
  LAMBDA_L1=100.0
  LAMBDA_L1_B=60.0
  LAMBDA_CX=0.0
  LAMBDA_CX_B=0.0
  LAMBDA_L2=0.0
  LAMBDA_PATCH=0.000000
  LAMBDA_LOCAL_D=0.1
  LAMBDA_LOCAL_STYLE=0.0
  LAMBDA_SECOND=10.0
  DATASET_MODE='unpaired_few_fusion'
  USE_ATTENTION='--use_attention'
  WHERE_ADD='all'
  CONDITIONAL_D='--conditional_D'
  CONTINUE_TRAIN='--continue_train'
  VALIDATE_FREQ=50
  BLACK_EPOCH=0
  DISPLAY_FREQ=32
  PRINT_FREQ=32
  LR=0.00002
  ;;
'base_gray_texture_tx')
  MODEL='dualnet'
  DIRECTION='AtoC' # 'AtoB' or 'BtoC'
  NENCODE=4
  BATCH_SIZE=20
  VALIDATE_FREQ=10
  BLACK_EPOCH=0
  DISPLAY_FREQ=32
  PRINT_FREQ=32
  LR=0.0001
  ;;
'11018')
  MODEL='dualnet'
  DIRECTION='AtoC' # 'AtoB' or 'BtoC'
  NENCODE=4
  BATCH_SIZE=4
  LOAD_SIZE=64
  FINE_SIZE=64
  RESIZE_OR_CROP='none'
  NO_FLIP='--no_flip'
  NITER=200
  NITER_DECAY=800
  SAVE_EPOCH=50
  NITER=500
  NITER_DECAY=2500
  SAVE_EPOCH=10
  NEF=64
  NGF=32
  NDF=32
  NET_G='dualnet'
  NET_D='basic_64'
  NET_D2='basic_64'
  NET_R='basic_64'
  NET_E='resnet_64'
  LAMBDA_L1=0.0
  LAMBDA_L1_B=60.0
  LAMBDA_GARY=100.0
  LAMBDA_CX=0.0
  LAMBDA_CX_B=0.0
  LAMBDA_TX=10.0
  LAMBDA_TX_B=5.0
  LAMBDA_L2=0.0
  LAMBDA_PATCH=0.00001
  DATASET_MODE='unpaired_few_fusion'
  USE_ATTENTION='--use_attention'
  WHERE_ADD='all'
  CONDITIONAL_D='--conditional_D'
  CONTINUE_TRAIN='--continue_train'
  VALIDATE_FREQ=10
  BLACK_EPOCH=0
  DISPLAY_FREQ=4
  PRINT_FREQ=4
  LR=0.0001
  ;;
'11030')
  MODEL='dualnet'
  DIRECTION='AtoC' # 'AtoB' or 'BtoC'
  NENCODE=4
  BATCH_SIZE=4
  LOAD_SIZE=64
  FINE_SIZE=64
  RESIZE_OR_CROP='none'
  NO_FLIP='--no_flip'
  NITER=500
  NITER_DECAY=2500
  SAVE_EPOCH=10
  NEF=64
  NGF=32
  NDF=32
  NET_G='dualnet'
  NET_D='basic_64'
  NET_D2='basic_64'
  NET_R='basic_64'
  NET_E='resnet_64'
  LAMBDA_L1=0.0
  LAMBDA_L1_B=60.0
  LAMBDA_GARY=100.0
  LAMBDA_CX=0.0
  LAMBDA_CX_B=0.0
  LAMBDA_L2=0.0
  LAMBDA_PATCH=0.00001
  DATASET_MODE='unpaired_few_fusion'
  USE_ATTENTION='--use_attention'
  WHERE_ADD='all'
  CONDITIONAL_D='--conditional_D'
  CONTINUE_TRAIN='--continue_train'
  VALIDATE_FREQ=10
  BLACK_EPOCH=0
  DISPLAY_FREQ=4
  PRINT_FREQ=4
  LR=0.0001
  ;;
'11027')
  MODEL='dualnet'
  DIRECTION='AtoC' # 'AtoB' or 'BtoC'
  NENCODE=4
  BATCH_SIZE=4
  LOAD_SIZE=64
  FINE_SIZE=64
  RESIZE_OR_CROP='none'
  NO_FLIP='--no_flip'
  NITER=500
  NITER_DECAY=2500
  SAVE_EPOCH=10
  NEF=64
  NGF=32
  NDF=32
  NET_G='dualnet'
  NET_D='basic_64'
  NET_D2='basic_64'
  NET_R='basic_64'
  NET_E='resnet_64'

  LAMBDA_L1=0.0
  LAMBDA_L1_B=60.0
  LAMBDA_GARY=100.0
  LAMBDA_CX=0.0
  LAMBDA_CX_B=0.0
  LAMBDA_L2=0.0
  LAMBDA_PATCH=0.00001


  DATASET_MODE='unpaired_few_fusion'
  USE_ATTENTION='--use_attention'
  WHERE_ADD='all'
  CONDITIONAL_D='--conditional_D'
  CONTINUE_TRAIN='--continue_train'
  VALIDATE_FREQ=10
  BLACK_EPOCH=0
  DISPLAY_FREQ=4
  PRINT_FREQ=4
  LR=0.0001
  ;;
'skeleton_gray_color')
  MODEL='dualnet'
  DIRECTION='AtoC' # 'AtoB' or 'BtoC'
  NENCODE=10
  FEW_SIZE=30
  BATCH_SIZE=80
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
  BATCH_SIZE=80
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
  NET_DLOCAL='basic_32'
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

DATE=`date '+%d_%m_%Y-%H_conv1'`    # delete minute for more convinent continue training, just run one experiment in an hour
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
  --netD_local ${NET_DLOCAL} \
  --use_dropout \
  --dataset_mode ${DATASET_MODE} \
  --lambda_L1 ${LAMBDA_L1} \
  --lambda_L1_B ${LAMBDA_L1_B} \
  --lambda_TX ${LAMBDA_TX} \
  --lambda_TX_B ${LAMBDA_TX_B} \
  --lambda_CX ${LAMBDA_CX} \
  --lambda_CX_B ${LAMBDA_CX_B} \
  --lambda_L2 ${LAMBDA_L2} \
  --lambda_patch ${LAMBDA_PATCH} \
  --lambda_GAN ${LAMBDA_GAN} \
  --lambda_GAN_B ${LAMBDA_GAN_B} \
  --lambda_local_D ${LAMBDA_LOCAL_D} \
  --lambda_local_style ${LAMBDA_LOCAL_STYLE} \
  --lambda_second ${LAMBDA_SECOND} \
  --where_add ${WHERE_ADD} \
  ${CONDITIONAL_D} \
  ${CONTINUE_TRAIN} \
  --black_epoch_freq ${BLACK_EPOCH} \
  --validate_freq ${VALIDATE_FREQ} \
  --display_freq ${DISPLAY_FREQ} \
  --print_freq ${PRINT_FREQ} \
  --lr ${LR}


set -ex

MODEL='dualnet'
CLASS=${1}
GPU_ID=${2}


DISPLAY_ID=`date '+%H%M'`
# DISPLAY_ID=0

PORT=9999
# PORT=10000

NENCODE=4
FEW_SIZE=0 # no use for english dataset

CHECKPOINTS_DIR=checkpoints/${CLASS}/  # execute .sh in project root dir to ensure right path


# dataset
NO_FLIP='--no_flip'
DIRECTION='AtoC'
LOAD_SIZE=64
FINE_SIZE=64
RESIZE_OR_CROP='none'
NO_FLIP='--no_flip'
INPUT_NC=3
BATCH_SIZE=16
DATASET_MODE='multi_fusion'
WHERE_ADD='all'
CONDITIONAL_D=''

# Networks module
NGF=64
NDF=32
NEF=32

NET_G='dualnet'
NET_D='basic_64'
NET_D2='basic_64'
NET_DLOCAL='basic_32'

USE_ATTENTION='--use_attention'
CONDITIONAL_D='--conditional_D'

LR=0.0002


VALIDATE_FREQ=0
DISPLAY_FREQ=100

LAMBDA_L1=100.0
LAMBDA_L1_B=50.0
LAMBDA_CX=25.0
LAMBDA_CX_B=15.0
LAMBDA_GAN=1.0
LAMBDA_GAN_B=1.0
LAMBDA_LOCAL_D=1.0



# dataset parameters
case ${CLASS} in
'base_gray_color' | 'base_gray_color_s')
  NENCODE=4
  BATCH_SIZE=80

  NITER=50
  NITER_DECAY=250

  SAVE_EPOCH=2

  LAMBDA_L1=100.0
  LAMBDA_L1_B=50.0
  LAMBDA_CX=25.0
  LAMBDA_CX_B=15.0
  LAMBDA_GAN=1.0
  LAMBDA_GAN_B=1.0
  LAMBDA_LOCAL_D=1.0
  DATASET_MODE='multi_fusion'

  CONTINUE_TRAIN=''

  PRINT_FREQ=400
  ;;

'base_gray_texture' | 'base_gray_texture_s')
  DATA_ID=${3}     # 0-34 means train the id dataset, 35 means train all the 35 dataset
  CLASS=$CLASS'_'$DATA_ID

  NENCODE=4
  BATCH_SIZE=80

  NITER=500
  NITER_DECAY=2500

  SAVE_EPOCH=100

  LAMBDA_L1=100.0
  LAMBDA_L1_B=60.0
  LAMBDA_CX=50.0
  LAMBDA_CX_B=15.0
  LAMBDA_LOCAL_D=0.1
  DATASET_MODE='few_fusion'

  CONTINUE_TRAIN='--continue_train'
  VALIDATE_FREQ=50

  DISPLAY_FREQ=100
  LR=0.00002
  ;;

'base_gray_texture_unpaired' | 'base_gray_texture_unpaired_s')
  NENCODE=4
  BATCH_SIZE=32

  NITER=500
  NITER_DECAY=2500
  SAVE_EPOCH=10

  LAMBDA_GAN=1.0
  LAMBDA_GAN_B=1.0
  LAMBDA_LOCAL_D=0.1
  DATASET_MODE='few_fusion'

  CONTINUE_TRAIN='--continue_train'
  VALIDATE_FREQ=50

  DISPLAY_FREQ=32
  PRINT_FREQ=32
  LR=0.00002
  ;;

'skeleton_gray_color' | 'skeleton_gray_color_s')
  NENCODE=10
  FEW_SIZE=30
  BATCH_SIZE=80

  NITER=10
  NITER_DECAY=10
  SAVE_EPOCH=2
  LAMBDA_L1=100.0
  LAMBDA_L1_B=50.0
  LAMBDA_CX=25.0
  LAMBDA_CX_B=15.0

  DATASET_MODE='cn_multi_fusion'
  ;;

  'skeleton_gray_texture_s')
  NENCODE=10
  FEW_SIZE=30
  BATCH_SIZE=80

  NITER=40
  NITER_DECAY=60
  SAVE_EPOCH=10
  LAMBDA_L1=100.0
  LAMBDA_L1_B=20.0
  LAMBDA_CX=25.0
  LAMBDA_CX_B=15.0
  LAMBDA_L2=100.0

  DATASET_MODE='cn_multi_fusion'

  CONTINUE_TRAIN='--continue_train'

  DISPLAY_FREQ=100
  ;;
*)
  echo 'WRONG category: '${CLASS}
  exit
  ;;
esac

DATE=`date '+%d_%m_%Y-%H'`    # delete minute for more convinent continue training, just run one experiment in an hour
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
  --save_epoch_freq ${SAVE_EPOCH} \
  --input_nc ${INPUT_NC} \
  --niter ${NITER} \
  --niter_decay ${NITER_DECAY} \
  --ngf ${NGF} \
  --ndf ${NDF} \
  --nef ${NEF} \
  --netG ${NET_G} \
  --netD ${NET_D} \
  --netD_B ${NET_D2} \
  --netD_local ${NET_DLOCAL} \
  --use_dropout \
  --dataset_mode ${DATASET_MODE} \
  --lambda_L1 ${LAMBDA_L1} \
  --lambda_L1_B ${LAMBDA_L1_B} \
  --lambda_CX ${LAMBDA_CX} \
  --lambda_CX_B ${LAMBDA_CX_B} \
  --lambda_GAN ${LAMBDA_GAN} \
  --lambda_GAN_B ${LAMBDA_GAN_B} \
  --lambda_local_D ${LAMBDA_LOCAL_D} \
  ${CONDITIONAL_D} \
  ${CONTINUE_TRAIN} \
  --validate_freq ${VALIDATE_FREQ} \
  --display_freq ${DISPLAY_FREQ} \
  --print_freq ${PRINT_FREQ} \
  --lr ${LR}


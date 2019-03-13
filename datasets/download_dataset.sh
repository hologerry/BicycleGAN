FILE=$1

if [[ $FILE != "base_gray_color" && $FILE != "base_gray_texture" &&  $FILE != "skeleton_gray_color" && $FILE != "skeleton_gray_texture" ]]; then
  echo "Available datasets are base_gray_color, base_gray_texture, skeleton_gray_color and skeleton_gray_texture"
  exit 1
fi

echo "Specified [$FILE]"

URL=http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/$FILE.tar.gz
TAR_FILE=./datasets/$FILE.tar.gz
TARGET_DIR=./datasets/$FILE/
wget -N $URL -O $TAR_FILE
mkdir -p $TARGET_DIR
tar -zxvf $TAR_FILE -C ./datasets/
rm $TAR_FILE

#!/bin/bash

# This script downloads ImageNet-1k (ILSVRC2012) dataset
# You need to first register and get credentials from image-net.org

# Replace these with your ImageNet credentials
USERNAME="user"
PASSWORD="pass"

# Create directories
mkdir -p imagenet/train imagenet/validation

# Download train and validation archives
# Training images (138GB)
wget --user=$USERNAME --password=$PASSWORD https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar

# Validation images (6.3GB)
wget --user=$USERNAME --password=$PASSWORD https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar

# Download mapping files
wget --user=$USERNAME --password=$PASSWORD https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz

# Extract validation images
cd imagenet/validation
tar xvf ../../ILSVRC2012_img_val.tar
cd ../..

# Extract training images
cd imagenet/train
tar xvf ../../ILSVRC2012_img_train.tar

# Extract individual class archives
for f in *.tar; do
  d=`basename $f .tar`
  mkdir -p $d
  cd $d
  tar xvf ../$f
  cd ..
  rm $f
done

cd ../..

# Extract devkit
tar xvf ILSVRC2012_devkit_t12.tar.gz

# Cleanup
rm ILSVRC2012_img_train.tar ILSVRC2012_img_val.tar ILSVRC2012_devkit_t12.tar.gz

echo "Download and extraction complete!"

# Optional: verify number of images
echo "Verifying image counts..."
echo "Number of training images: $(find imagenet/train -type f | wc -l)"
echo "Number of validation images: $(find imagenet/validation -type f | wc -l)"

#!/bin/bash
# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Script to preprocess PlantVillage Challenge
# training and validation data set.
#
# The final output of this script are sharded TFRecord files containing
# serialized Example protocol buffers. See build_plantvillage_data.py for
# details of how the Example protocol buffers contain the PlantVillage data.
#
# The final output of this script appears as such:
#
#   data_dir/train-00000-of-01024
#   data_dir/train-00001-of-01024
#    ...
#   data_dir/train-00127-of-01024
#
# and
#
#   data_dir/validation-00000-of-00128
#   data_dir/validation-00001-of-00128
#   ...
#   data_dir/validation-00127-of-00128
#
# Note that this script may take several hours to run to completion. The
# conversion of the ImageNet data to TFRecords alone takes 2-3 hours depending
# on the speed of your machine. Please be patient.
#
# **IMPORTANT**
# To download the raw images, the user must create an account with image-net.org
# and generate a username and access_key. The latter two are required for
# downloading the raw images.
#
# usage:
#  ./preprocess_plantvillage.sh [data-dir]
set -e

if [ -z "$1" ]; then
  echo "usage preprocess_plantvillage.sh [data dir]"
  exit
fi

# Create the output and temporary directories.
DATA_DIR="${1%/}"
SCRATCH_DIR="${DATA_DIR}/raw-data/"
CROWDAI_DIR="${DATA_DIR}/crowdai/"
mkdir -p "${DATA_DIR}"
mkdir -p "${SCRATCH_DIR}"
WORK_DIR="$0.runfiles/__main__/inception"
CURRENT_DIR=$(pwd)

# Note the locations of the train and validation data.
TRAIN_DIRECTORY="${SCRATCH_DIR}train/"
VALIDATION_DIRECTORY="${SCRATCH_DIR}validation/"

if [ ! -d ${TRAIN_DIRECTORY} ]; then
	cp -R ${CROWDAI_DIR} ${TRAIN_DIRECTORY}
fi

# Generate a list of labels
LABELS_FILE="${DATA_DIR}/labels.txt"
ls -1 "${TRAIN_DIRECTORY}" | grep -v 'LICENSE' | sed 's/\///' | sort > "${LABELS_FILE}"

# Generate the validation data set.
if [ ! -d ${VALIDATION_DIRECTORY} ]; then
	while read LABEL; do
	  VALIDATION_DIR_FOR_LABEL="${VALIDATION_DIRECTORY}${LABEL}"
	  TRAIN_DIR_FOR_LABEL="${TRAIN_DIRECTORY}${LABEL}"

	  # Move the first randomly selected 100 images to the validation set.
	  mkdir -p "${VALIDATION_DIR_FOR_LABEL}"
	  VALIDATION_IMAGES=$(ls -1 "${TRAIN_DIR_FOR_LABEL}" | shuf | head -100)
	  for IMAGE in ${VALIDATION_IMAGES}; do
	    mv -f "${TRAIN_DIRECTORY}${LABEL}/${IMAGE}" "${VALIDATION_DIR_FOR_LABEL}"
	  done
	done < "${LABELS_FILE}"
fi

# Build the TFRecords version of the image data.
cd "${CURRENT_DIR}"
BUILD_SCRIPT="${WORK_DIR}/build_image_data"
echo $BUILD_SCRIPT
OUTPUT_DIRECTORY="${DATA_DIR}"
"${BUILD_SCRIPT}" \
  --train_directory="${TRAIN_DIRECTORY}" \
  --validation_directory="${VALIDATION_DIRECTORY}" \
  --output_directory="${OUTPUT_DIRECTORY}" \
  --labels_file="${LABELS_FILE}"

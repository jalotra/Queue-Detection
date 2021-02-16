#!/usr/bin/env bash

# This script setups the development server
# Requirements  : conda or miniconda

# I am using Yolo from Pytorch Hub. 
# See this : https://github.com/ultralytics/yolov5/issues/36

# Step 1 
ENV_NAME="QUEUE_DETECTION"
conda init bash
conda activate $ENV_NAME

# Step 2 : Install Dependencies of UltraLytics 
pip install -U opencv-python pillow pyYAML tqdm

# Step 3 : Install Pytorch 
function get_cuda_version(){
    echo "Since you pressed 'Yes', what is the cuda version listed in /usr/local/cuda/version.txt ?"
    echo "Options are '9.2' '10.1' '10.2' '11.0'"
    select yn in "9.2" "10.1" "10.2" "11.0"; do 
        9.2 ) conda install pytorch torchvision torchaudio cudatoolkit=9.2 -c pytorch;;
        10.1 ) conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch;;
        10.2 ) conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch;;
        11.0 ) conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch;;
        * ) echo "You fucked up !"; exit;; 
    esac 
done
}



function has_gpu () {
    echo "Do You have a GPU and more importantly CUDA installed." ? 
    echo "Run cat /usr/local/cuda/version.txt"
    echo "Type Yes or No, Make sure this is Case-Sensitive"
    select yn in "Yes" "No"; do
    case $yn in
        Yes ) get_cuda_version();;
        No ) conda install pytorch torchvision torchaudio cpuonly -c pytorch;;
    esac
done
}


has_gpu()



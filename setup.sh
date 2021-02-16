#!/usr/bin/env bash

# This script setups the development server
# Requirements  : conda or miniconda

# I am using Yolo from Pytorch Hub. 
# See this : https://github.com/ultralytics/yolov5/issues/36

# Step 1 : Install Dependencies of UltraLytics 
pip install -U opencv-python pillow pyYAML tqdm
pip install -r requirements.txt

# Step 2 : Install Pytorch 
function get_cuda_version(){
    echo -e "Since you pressed 'Yes', what is the cuda version listed in /usr/local/cuda/version.txt ? \n"
    echo -e "Options are '9.2' '10.1' '10.2' '11.0' \n"
    while true; do
    read -p "Type Yes or No, Make sure this is Case-Sensitive. " yn
    case $yn in 
        9.2 ) conda install pytorch torchvision torchaudio cudatoolkit=9.2 -c pytorch; break;;
        10.1 ) conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch; break;;
        10.2 ) conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch; break;;
        11.0 ) conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch; break;;
        * ) echo "You messed up !"; break;; 
    esac 
done
}


function has_gpu () {
    echo "Do You have a GPU and more importantly CUDA installed." ? 
    echo "Run cat /usr/local/cuda/version.txt"
    while true; do
    read -p "Type Yes or No, Make sure this is Case-Sensitive. " yn
    case $yn in
        [Yy]* ) get_cuda_version; break;;
        [Nn]* ) conda install pytorch torchvision torchaudio cpuonly -c pytorch; break;;
        * ) echo "Please answer yes or no.";;
    esac
done

}

has_gpu



# Инструкция для настройки рабочего окружения для работы с computer vision.

## 1. Установка зависимостей для компиляции библиотек.

```bash
sudo apt update -y; apt install -y \
        git \
        wget \
        cmake \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libopenblas-dev \
        liblapack-dev \
        curl \
        python3.8-dev \
        python3.8-distutils \
        build-essential \
        cmake \
        unzip \
        pkg-config \
        libtbb-dev \
        libgtk-3-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libv4l-dev \
        libxvidcore-dev \
        libx264-dev \
        libatlas-base-dev \
        gfortran
```
Установка Gstreamer
```bash
apt-get install -y libgstreamer1.0-0 \
            gstreamer1.0-plugins-base \
            gstreamer1.0-plugins-good \
            gstreamer1.0-plugins-bad \
            gstreamer1.0-plugins-ugly \
            gstreamer1.0-libav \
            gstreamer1.0-doc \
            gstreamer1.0-tools \
            libgstreamer1.0-dev \
            libgstreamer-plugins-base1.0-dev
```


## 2. Создание виртуальной среды.

Скачать pip и установить:
``` bash
wget https://bootstrap.pypa.io/get-pip.py
sudo python3 get-pip.py
```

Установить virtualenv и virtualenvwrapper
```bash
sudo pip install virtualenv virtualenvwrapper
```

Прописать в ~/.bashrc следующие строки и затем обновить терминал source ~/.bashrc:
```bash
export WORKON_HOME=$HOME/.virtualenvs
export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
source /usr/local/bin/virtualenvwrapper.sh
```
Создать виртуальную среду c названием cv (любое название) и активировать:
```bash
mkvirtualenv cv -p python3.8
workon cv
```
Устанавливаем numpy
```bash
pip install numpy
```
Все дальнейшие действия должны выполняться в активированной виртуальной среде!

## 3. Установка Cuda 11.1
``` bash
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
    sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
    sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
    sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
    sudo apt-get update
    sudo apt-get -y install cuda-11-1
```
Добавить переменные среды в ~/.bashrc
``` bash
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH
```

Перезагрузить компьютер и проверить установку
``` bash
nvcc --version
```
## 4. Установка CUDNN 8.2.0
Скачать cudnn 8.2.0

https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.2.0.53/11.3_04222021/cudnn-11.3-linux-x64-v8.2.0.53.tgz

Распаковать
``` bash
tar -xzvf cudnn-11.3-linux-x64-v8.2.0.53.tgz
```

Скопируйте следующие файлы в каталог CUDA Toolkit .
``` bash
sudo cp cuda/include/cudnn*.h /usr/local/cuda/include 
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64 
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

## 5. Установка TensorRT
Скачиваем и распаковываем .tar архив
https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/7.2.3/tars/TensorRT-7.2.3.4.Ubuntu-18.04.x86_64-gnu.cuda-11.1.cudnn8.1.tar.gz

Устанавливаем tensorrt
```bash
cd TensorRT-7.2.3.4
cd python
pip install tensorrt-7.2.3.4-cp38-none-linux_x86_64.whl
cd ../uff
pip install uff-0.6.9-py2.py3-none-any.wh
cd ../graphsurgeon/
pip install graphsurgeon-0.4.5-py2.py3-none-any.whl
cd ../onnx_graphsurgeon
pip install onnx_graphsurgeon-0.2.6-py2.py3-none-any.whl
```

Добавить переменные среды в ~/.bashrc
``` bash
export TENSORRT_HOME=~/TensorRT-7.2.3.4
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TENSORRT_HOME/lib
export PATH=$PATH:$TENSORRT_HOME/bin
```


## 6. Установка NVIDIA CODEC SDK
Скачать NVIDIA CODEC SDK 
https://developer.nvidia.com/video_codec_sdk/downloads/v11

Установить
```bash
unzip Video_Codec_SDK_11.0.10.zip
sudo cp Video_Codec_SDK_11.0.10/Lib/linux/stubs/x86_64/* /usr/local/cuda/lib64/stubs/
sudo cp Video_Codec_SDK_11.0.10/Interface/* /usr/local/cuda/include
```

## 7. Сборка onnx 
```bash
sudo apt-get install protobuf-compiler libprotoc-dev
git clone https://github.com/onnx/onnx.git
cd onnx
git checkout v1.9.0
git submodule update --init --recursive
python setup.py install
```

## 8. Сборка onnxruntime

Пока лучше ставить из pip пакета:
```bash
pip install onnxruntime-gpu==1.7.0
```

Сборка с поддержкой CUDA 
```bash
git clone https://github.com/microsoft/onnxruntime.git
cd onnxruntime
./build.sh --config Release --update --build --parallel --build_wheel \
 --use_cuda --cuda_home $CUDA_HOME --cudnn_home $CUDA_HOME 
```
Сборка с поддержкой TensorRT 
```bash
./build.sh --config Release --update --build --parallel --build_wheel \
 --use_tensorrt --cuda_home $CUDA_HOME --cudnn_home $CUDA_HOME \
 --tensorrt_home $TENSORRT_HOME
```
Установка собранного .whl файла
```bash
 cd build/Linux/Release/dist
 pip install onnxruntime_gpu_tensorrt-1.7.0-cp38-cp38-linux_x86_64.whl
 ```

## 9. Сборка opencv

Задаём переменную окружения с версией Opencv

``` bash
export OPENCV_VERSION=4.5.2
```
Задаём переменную окружения для compute capability GPU.
Значения compute capability для видеокарт можно посмотреть тут:
https://developer.nvidia.com/cuda-gpus

``` bash
export COMPUTE_CAPABILITY="6.1"
```
Скачиваем и распаковываем Opencv
```bash
wget -O opencv.zip https://github.com/opencv/opencv/archive/$OPENCV_VERSION.zip
unzip opencv.zip
mv opencv-$OPENCV_VERSION opencv

wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/$OPENCV_VERSION.zip
unzip opencv_contrib.zip
mv opencv_contrib-$OPENCV_VERSION opencv_contrib
```
Собираем Opencv
``` bash
mkdir build && cd build

cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=~/opencv_bin \
    -D WITH_TBB=ON \
    -D WITH_CUDA=ON \
    -D WITH_CUDNN=ON \
    -D OPENCV_DNN_CUDA=ON \
    -D CUDA_ARCH_BIN=$COMPUTE_CAPABILITY \
    -D ENABLE_FAST_MATH=ON \
    -D CUDA_FAST_MATH=ON \
    -D WITH_CUBLAS=ON \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D OPENCV_PYTHON3_INSTALL_PATH=$(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
    -D WITH_V4L=ON \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D PYTHON_DEFAULT_EXECUTABLE=$(which python) \
    -D PYTHON_EXECUTABLE=$(which python) \
    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D WITH_NVCUVID=ON \
    -D WITH_NVDECODE=ON \
    -D WITH_NVENCODE=ON \
    -D WITH_GSTREAMER=ON \
    -D WITH_OPENCL=OFF \
    -D BUILD_EXAMPLES=OFF ..

make -j"$(nproc)"
make install
sudo ldconfig
```

## 10. Сборка PyTorch
``` bash
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
git checkout v1.8.1
git submodule sync
git submodule update --init --recursive
pip install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses typing
sudo apt-get install libomp-dev
python setup.py install
```
## 11. Сборка Torchvision
```bash
git clone https://github.com/pytorch/vision.git
cd vision
git checkout v0.9.1
pip uninstall pillow
CC="cc -mavx2" pip install -U --force-reinstall pillow-simd
python setup.py install
```

## 12. Установка DLIB

```bash
git clone -b 'v19.22' --single-branch https://github.com/davisking/dlib.git
cd dlib
python setup.py install
```

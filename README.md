# Installation Steps

**Note: I am using Ubuntu 16.04 LTS OS on 64 bit machine and without any GPU.**

* Clone TensorFlow Object detection repository.

```bash
cd ~
git clone https://github.com/AmitChotaliya/video_object_detection.git
cd video_object_detection
git clone https://github.com/tensorflow/models.git
```

* Install virtualenv using pip

```bash
pip3 install virtualenv
```

* Create python3 virtual environment

```bash
cd ~/video_object_detection
virtualenv -p python3 .
source bin/activate
```

* Install  all the required packages

```bash
pip install -r requirements.txt
pip install tensorflow
sudo apt-get install protobuf-compiler python-pil python-lxml python-tk
pip install Cython
pip install contextlib2
pip install jupyter
pip install matplotlib
pip install pillow
pip install lxml
pip install ffmpeg-python
cd models/research
wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.6.1/protoc-3.6.1-linux-x86_64.zip
unzip protobuf.zip
# From tensorflow/models/research/
./bin/protoc object_detection/protos/*.proto --python_out=.
pip install opencv-python
```

* Copy the ods.py in object_detection module

```bash
cp ods.py models/research/object_detection/
```


# Usage

* Run the following command with first argument as mp4 file path.

```bash
cd models/research/object_detection/
python ods.py /tmp/test2.mp4
```
* It will process the video and store the processed avi in `/tmp` directory

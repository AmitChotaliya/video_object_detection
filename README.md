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
python3 -m venv .
source bin/activate
```

* Install  all the required packages

```bash
pip install -r requirements.txt
```

# Usage

* Run the following command with first argument as mp4 file path.

```bash
python ods.py /tmp/test2.mp4
```
* It will process the video and store the processed avi in `/tmp` directory
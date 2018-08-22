import sys
import time
import numpy as np
import tensorflow as tf
import cv2
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import ffmpeg
import zipfile

from collections import defaultdict
from io import StringIO
from PIL import Image
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

print(tf.__version__)

# %matplotlib inline

import matplotlib; matplotlib.use('Agg')
from utils import label_map_util
from utils import visualization_utils as vis_util


# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2018_01_28'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90


if not os.path.isdir('ssd_mobilenet_v1_coco_2018_01_28'):
    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd())

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# https://stackoverflow.com/questions/42798634/extracting-keyframes-python-opencv
# pip install -r requirements.txt

def load_image_into_numpy_array(image):
    print(image)
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

#https://stackoverflow.com/questions/10559035/how-to-rotate-a-video-with-opencv

def anaylize_video(video_path):
    cap = cv2.VideoCapture(video_path)
    output_file_path = "/tmp/"+os.path.splitext(os.path.basename(video_path))[0]+".avi"
    out = None
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    org_fps = cap.get(cv2.CAP_PROP_FPS)
    length2 = length
    previousFrame = None
    print('Total Frames = '+str(length))

    while length:
        length -= 1
        print('Processing Frame = '+str(length))
        ret, image = cap.read()
        if ret == 0:
            break

        if out is None:
            [h, w] = image.shape[:2]
            out = cv2.VideoWriter(output_file_path, 0, org_fps, (w, h))



        image_np = image
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        start_time = time.time()
        output_dict = run_inference_for_single_image(image_np, detection_graph)
        elapsed_time = time.time() - start_time
        print('inference time cost: {}'.format(elapsed_time))
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=8)
        out.write(image_np)
    print("Closing everything")
    cap.release()
    out.release()
    return output_file_path


def convert_video(avi_file, source_file_path):
    source_file_dir = os.path.dirname(source_file_path)
    source_file_name = os.path.splitext(os.path.basename(source_file_path))[0]
    final_file_path = source_file_dir + "/" + source_file_name + ".mp4"
    stream = ffmpeg.input(avi_file)
    # stream = ffmpeg.output(stream, source_file_dir+"/"+source_file_name+".mp4", vcodec='libx264', metadata='s:v rotate="0"', vf="transpose=3", crf=23, acodec="copy")
    stream = ffmpeg.output(stream, final_file_path, vcodec='libx264', crf=23, acodec="copy")
    stream = ffmpeg.overwrite_output(stream)
    ffmpeg.run(stream)
    return final_file_path

def do_cleanup(avi_file, source_file):
    try:
        os.remove(avi_file)
        os.remove(source_file)
    except OSError:
        pass



def find_avi(dir):
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(".avi"):
                file_path = os.path.join(root, file)
                return file_path, root
    return None, None

def lock_file(file_path):
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    source_file_dir = os.path.dirname(file_path)
    locked_file_path = source_file_dir+"/"+file_name+".lock"
    os.rename(file_path, locked_file_path)
    return locked_file_path

def get_running_ods_procs():
  import subprocess
  count = 0
  try:
      output = subprocess.check_output("pgrep -cf ods.py",shell=True,stderr=subprocess.STDOUT)
      count = int(output.rstrip())
      print("Runnin procs are = "+str(count))
      return count
  except subprocess.CalledProcessError as e:
      raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
  return count

if __name__ == "__main__":
    import sys
    import subprocess
    file_id = None
    org_file = None

    current_proc_count = get_running_ods_procs()
    print("Running processes"+str(current_proc_count))
    if current_proc_count > 2:
        print("Max number of processes already running")
        exit()

    try:
        file_id = sys.argv[1]
    except IndexError:
        pass


    if not file_id:
        file_id, root = find_avi('/data/video-share/media')
        org_file = file_id
        if not file_id:
           print("No pending files found")
           exit()
        print("Found file "+file_id)
        file_id = lock_file(file_id)
        print("Locked file ID = "+file_id)

    file_name = os.path.basename(file_id)
    output_file = anaylize_video(file_id)
    final_file_path = convert_video(output_file, file_id)
    do_cleanup(file_id, output_file)    

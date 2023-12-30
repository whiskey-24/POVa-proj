# Detection, tracking and map registration of vehicles in top-view images
## POVa project 

### Project description

Project consists of three main parts:
1. Detection of vehicles in top-view images
2. Tracking of vehicles in top-view images
3. Map registration of vehicles in top-view images

Each part was developed independently and can be used separately.

### GitHub link

https://github.com/whiskey-24/POVa-proj

### Project structure

Detection part:
- `misc` - miscellaneous files. Mainly used for converting dataset files 
between different formats (for different neural network training frameworks)
- - `convert_to_YOLO_OBB.py` - converts dataset files from p_neuma format to 
YOLO OBB format
- - `convert_to_yolov5_OBB_format.py` - converts dataset files from p_neuma 
format to YOLOv5 OBB format
- - `divide_frames.py` - crops frames from p_neuma dataset to multiple subframes
along with corresponding annotations
- - `file_purger.sh` - removes cropped frames from desired directory
- - `retinaface_to_YOLO.py` - converts retinaface annotations to YOLO format
- `yolov8` - YOLOv8 implementation based on https://github.com/ultralytics/ultralytics
- - `*.yaml` - configuration files for training and testing
- - `train_yolo.py` - training script
- - `test_yolo.py` - testing script
- - `merge_datasets.sh` - script for merging multiple YOLO format datasets into one
- `retinaface` - RetinaFace implementation based on https://github.com/biubug6/Pytorch_Retinaface
- - `evaluate_retinaface.py` - evaluation script
- - `Pytorch_Retinaface` - RetinaFace implementation cloned from github. Only changed
file is `detect.py` that is used for inference and has been modified to output
Vehicle objects. Also, there were minor changes to number of files to allow for 
multi GPU training.
- `p_neuma` - scripts for manipulating p_neuma dataset. Cloned from https://github.com/shgold/pNEUMA-Vision-toolbox  

Tracking part:
- - `src/deep_sort_rt.py` - integrates DeepSort real time for evaluation on p_neuma
- - `src/deep_sort.py` - integrates DeepSort for evaluation on p_neuma
- - `src/strong_sort.py` - integrates StrongSort for evaluation on p_neuma
- - `src/utils_tracker` - various functions for manipulating with p_neuma dataset
- - `src/vehicle.py` - vehicle detection and tracking classes for tracking evaluation
- `try_trackers.py` - showcases the use of different trackers and evaluates them

Map registration part:
- `satellite_extractor` - satellite images extractor based on https://github.com/Jimut123/jimutmap
- - `jimutmap` - satellite images extractor cloned from github. In its original form
it was broken and had to be fixed. The multi-threading part needed to be changed along
with URL generation. Also tiling functionality was non-salvageable and had to be
reimplemented.
- - `satellite_extractor.py` - script for extracting satellite images using jimutmap.
With custom tiling functionality and coordinates computation.
- `point_reg` - point registration based on SuperGlue and SuperPoint
- - `SuperGlueModel` - SuperGlue implementation cloned from https://github.com/magicleap/SuperGluePretrainedNetwork
- - `super_glue.py` - script implementing SuperGlue, image transformation, 
homography calculation and pixel coordinates conversion.

Root directory:
- `application.py` - main script for running the application
- `custom_dataclasses.py` - custom dataclasses used throughout the project
- `work_todos.md` - list of tasks to be done
- `README.md` - this file

### Usage

Run `application.py` with desired inputs. You need to provide path to the 
video with top-view images and approximate latitude and longitude of the
camera that was used to record the video. You can also provide thresholds and
another parameters for detection and tracking.
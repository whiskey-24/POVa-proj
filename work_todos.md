## Detection

1. [ ] Download and run YOLOv8.
In case of any problems try running older versions.
2. [ ] Rewrite annotations from dataset into YOLO format
3. [ ] Train YOLOv8 on dataset
4. [ ] Evaluate YOLOv8 on dataset

YOLOv8: https://github.com/ultralytics/ultralytics  
Dataset: https://zenodo.org/records/7426506  
pNEUMA (dataset) website: https://open-traffic.epfl.ch/  
Scripts for extracting bbox from annotations: https://github.com/shgold/pNEUMA-Vision-toolbox  
YOLO format: https://docs.ultralytics.com/datasets/detect/#ultralytics-yolo-format  


## Tracking

1. [ ] Download and run DeepSORT/SORT
2. [ ] Use annotations from dataset as ground truths for tracking
3. [ ] Evaluate DeepSORT/SORT on dataset
4. [ ] Experiment with additional trackers like Kalman filter and model based
trackers (that have idea about feasible future positions of car)
5. [ ] Evaluate additional trackers on dataset


DeepSORT: https://github.com/nwojke/deep_sort  
DeepSORT realtime (install through pip): https://github.com/levan92/deep_sort_realtime  
Dataset: https://zenodo.org/records/7426506  
StrongSORT: https://github.com/bharath5673/StrongSORT-YOLO  
Kalman filter in opencv: https://pieriantraining.com/kalman-filter-opencv-python-example/ 
https://learnopencv.com/object-tracking-and-reidentification-with-fairmot/  
Moving Vehicle Tracking with a Moving Drone Based on
Track Association: https://www.mdpi.com/2076-3417/11/9/4046


## Registration in map

- [x] Download and run satellite images extractor
- [ ] Evaluate registration based on SIFT/SURF
- [ ] Evaluate registration based on SuperPoint/AdaLAM/SuperGlue
- [x] Translate any point on source image to point on map (GPS coordinates)


- [ ] Extract road network info from OpenStreetMap
- [ ] Segment roads from source images, skeletonize them and match them to
road network from OpenStreetMap
- [ ] Evaluate registration based on road network


Satellite images extractor: https://github.com/Jimut123/jimutmap  
Satellite image registration: https://github.com/satellite-image-deep-learning/techniques#31-image-registration  
Accessing OSM Data in Python: https://pygis.io/docs/d_access_osm.html

SIFT in Opencv: https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html  
SuperPoint: https://github.com/rpautrat/SuperPoint  
AdaLAM: https://github.com/cavalli1234/AdaLAM  
SuperGlue: https://github.com/magicleap/SuperGluePretrainedNetwork

Scikit-image skeletonize: https://scikit-image.org/docs/dev/auto_examples/edges/plot_skeleton.html


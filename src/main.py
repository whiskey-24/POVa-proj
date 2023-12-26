from deep_sort_rt import run_deep_sort_rt
from deep_sort import run_deep_sort
from strong_sort import run_strong_sort
from utils import load_dataset,create_video,evaluate_tracks
from deep_sort_realtime.deepsort_tracker import DeepSort
from deepsort import DeepSortTracker
from strongsort import StrongSORT
import cv2
import matplotlib.pyplot as plt
import argparse
import os

def main():

    parser = argparse.ArgumentParser(description='Run different versions of Deep SORT.')
    parser.add_argument('--version', type=int, choices=[1, 2, 3], help='1 for standard Deep SORT, 2 for Deep SORT Real Time, 3 for bboxes calculated from positions')

    args = parser.parse_args()


    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_script_dir)))
    dataset_path = os.path.join(parent_dir, "dataset")

    if args.version == 1:
           
        """
        This is the multi-target tracker.

        Parameters
        ----------
        metric : nn_matching.NearestNeighborDistanceMetric
            A distance metric for measurement-to-track association.
        max_age : int
            Maximum number of missed misses before a track is deleted.
        n_init : int
            Number of consecutive detections before the track is confirmed. The
            track state is set to `Deleted` if a miss occurs within the first
            `n_init` frames.

        Attributes
        ----------
        metric : nn_matching.NearestNeighborDistanceMetric
            The distance metric used for measurement to track association.
        max_age : int
            Maximum number of missed misses before a track is deleted.
        n_init : int
            Number of frames that a track remains in initialization phase.
        kf : kalman_filter.KalmanFilter
            A Kalman filter to filter target trajectories in image space.
        tracks : List[Track]
            The list of active tracks at the current time step.

        """
        tracker = DeepSortTracker(metric_name="euclidean", max_iou_distance=0.4, max_age=40, n_init=3, max_dist=0.3, nn_budget=200)
        #tracker = DeepSortTracker()
        processed_frames = []
        for frame, annotation in load_dataset(dataset_path):
            subset = annotation[annotation['ID'] > 100]
            sub = subset[subset['ID'] < 200]
            cars = annotation[annotation['Type'] == 'Car']
            processed_frame = run_deep_sort(tracker, frame, annotation)
            processed_frames.append(processed_frame)
            '''
            cv2.imshow('Frame', processed_frame)

            if cv2.waitKey(0) & 0xFF == ord('q'):
                break'''
        create_video(processed_frames, output_file='deep_sort.mp4', fps=5.0)
        #cv2.destroyAllWindows()


    elif args.version == 2:

        tracker = DeepSort(
        max_iou_distance=0.4,    #adjusted for similarity in vehicle shapes/sizes
        max_age=40,              #slightly higher to account for potential occlusions
        n_init=3,                #standard initialization phase length
        nms_max_overlap=0.9,     #higher as overlap is less likely in aerial views
        max_cosine_distance=0.3, #balancing appearance variations
        nn_budget=200,           #if computational resources allow, for better accuracy
        gating_only_position=True, #considering only position for aerial tracking
        override_track_class=None, #default setting
        embedder="mobilenet",    #good balance between performance and efficiency
        half=True,               #use half precision if GPU supports it
        bgr=True,                #match the color format of your input frames
        embedder_gpu=True,       #utilize GPU for embedder
        embedder_model_name=None, #default setting for MobileNet
        embedder_wts=None,       #default weights
        polygon=True,            #if bboxes are oriented
        today=None               #default setting
    )
        tracker = DeepSort()
        """

        Parameters
        ----------
        max_iou_distance : Optional[float] = 0.7
            Gating threshold on IoU. Associations with cost larger than this value are
            disregarded. Argument for deep_sort_realtime.deep_sort.tracker.Tracker.
        max_age : Optional[int] = 30
            Maximum number of missed misses before a track is deleted. Argument for deep_sort_realtime.deep_sort.tracker.Tracker.
        n_init : int
            Number of frames that a track remains in initialization phase. Defaults to 3. Argument for deep_sort_realtime.deep_sort.tracker.Tracker.
        nms_max_overlap : Optional[float] = 1.0
            Non-maxima suppression threshold: Maximum detection overlap, if is 1.0, nms will be disabled
        max_cosine_distance : Optional[float] = 0.2
            Gating threshold for cosine distance
        nn_budget :  Optional[int] = None
            Maximum size of the appearance descriptors, if None, no budget is enforced
        gating_only_position : Optional[bool]
            Used during gating, comparing KF predicted and measured states. If True, only the x, y position of the state distribution is considered during gating. Defaults to False, where x,y, aspect ratio and height will be considered.
        override_track_class : Optional[object] = None
            Giving this will override default Track class, this must inherit Track. Argument for deep_sort_realtime.deep_sort.tracker.Tracker.
        embedder : Optional[str] = 'mobilenet'
            Whether to use in-built embedder or not. If None, then embeddings must be given during update.
            Choice of ['mobilenet', 'torchreid', 'clip_RN50', 'clip_RN101', 'clip_RN50x4', 'clip_RN50x16', 'clip_ViT-B/32', 'clip_ViT-B/16']
        half : Optional[bool] = True
            Whether to use half precision for deep embedder (applicable for mobilenet only)
        bgr : Optional[bool] = True
            Whether frame given to embedder is expected to be BGR or not (RGB)
        embedder_gpu: Optional[bool] = True
            Whether embedder uses gpu or not
        embedder_model_name: Optional[str] = None
            Only used when embedder=='torchreid'. This provides which model to use within torchreid library. Check out torchreid's model zoo.
        embedder_wts: Optional[str] = None
            Optional specification of path to embedder's model weights. Will default to looking for weights in `deep_sort_realtime/embedder/weights`. If deep_sort_realtime is installed as a package and CLIP models is used as embedder, best to provide path.
        polygon: Optional[bool] = False
            Whether detections are polygons (e.g. oriented bounding boxes)
        today: Optional[datetime.date]
            Provide today's date, for naming of tracks. Argument for deep_sort_realtime.deep_sort.tracker.Tracker.
        """

        processed_frames = []
        vehicle_tracks = {}
        vehicle_detections_gt = {}
        for frame, annotation in load_dataset(dataset_path):

            subset = annotation[annotation['ID'] > 0]
            sub = subset[subset['ID'] < 31]
            cars = annotation[annotation['Type'] == 'Car']
            processed_frame = run_deep_sort_rt(tracker, frame, sub, vehicle_tracks, vehicle_detections_gt)
            processed_frames.append(processed_frame)
            '''
            cv2.imshow('Frame', processed_frame)

            if cv2.waitKey(0) & 0xFF == ord('q'):
                break'''
        create_video(processed_frames, output_file='deep_sort_rt.mp4', fps=5.0)
        #generate a text file with the tracks

        with open('tracks.txt', 'w') as f:
            for key in vehicle_tracks.keys():
                f.write("%s\n" % vehicle_tracks[key].type)
                f.write("%s\n" % vehicle_tracks[key].trajectory)
                f.write("%s\n" % vehicle_tracks[key].id)
                f.write("%s\n" % vehicle_tracks[key].original_ltwh)
                f.write("---------------------------------\n")

        f.close()

        with open('detections.txt', 'w') as f:
            for key in vehicle_detections_gt.keys():
                f.write("%s\n" % vehicle_detections_gt[key].type)
                f.write("%s\n" % vehicle_detections_gt[key].list_of_bboxes)
                f.write("%s\n" % vehicle_detections_gt[key].id)
                f.write("---------------------------------\n")

        #evaluate the tracks
        acc = evaluate_tracks(vehicle_tracks, vehicle_detections_gt, 0.5)
        print("Accuracy:", acc)
        #cv2.destroyAllWindows()
    elif args.version == 3:

        processed_frames = []
        for frame, annotation in load_dataset(dataset_path):
            for bbox, id in zip(annotation['ltwh'], annotation['ID']):
                left, top, width, height = bbox[0], bbox[1], bbox[2], bbox[3]
                right = left + width
                bottom = top + height
                cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (255, 255, 0), 2)
                cv2.putText(frame, str(id), (int(left), int(top)), 0, 5e-3 * 200, (255, 0, 255), 2)

                '''
                #cv2.imshow('Frame', frame)
                #if cv2.waitKey(0) & 0xFF == ord('q'):
                    break
        
            #cv2.destroyAllWindows()
            '''
            processed_frames.append(frame)
        create_video(processed_frames, output_file='from_dataset.mp4', fps=20.0)
    else:
        print("No version selected or invalid choice. Exiting.")
        return


if __name__ == '__main__':
    main()
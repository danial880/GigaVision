"""
usage: python gv/explore.py \
    --conf 0.3 \
    --height_ratio 0.15 \
    --width_ratio 0.05 \
    --IOS_thresh 0.7 \
    --model_path yolov7-e6e.pt \
    --model_type yolov7 \
    --path val/ \
    --annotations_path src/val.json \
    --csv src/training_data.csv \
    --outfile results.json
"""
import os
import sys
import json
import time
import torch
import logging
import argparse
import pandas as pd
import os.path as osp
import switch_ids as si
from tqdm import tqdm
from sahi import AutoDetectionModel
from search_space import SearchSpace
from metric import EvaluationMetrics
from sahi.predict import get_sliced_prediction


class Explore:
    """
    The Explore class contains the necessary methods and attributes for creating, processing, and
    storing the results of object detection with the YOLOv7 model.

    Attributes:
    conf (float): Confidence threshold for object detection predictions (default=0.25).
    height_ratio (float): Height ratio for overlapping slices of input images (default=0.2).
    width_ratio (float): Width ratio for overlapping slices of input images (default=0.1).
    IOS_thresh (float): Intersection Over Union threshold for post-processing of object detection
        predictions (default=0.8).
    category (list): List of object categories for predictions to be included in the final results
        (default=['person', 'bicycle','car', 'motorcycle', 'bus', 'truck']).
        yolov7_model_path (str): Path to the YOLOv7 model (default='yolov7-e6e.pt').
    path (str): Path to the input image directory (default='val/').
    annotations_path (str): Path to the annotations in JSON format (default='val.json').
    csv (str): Path to the training data in CSV format (default='training_data.csv').
    outfile (str): Filename for the output results in JSON format (default='space.json').
    data_csv (pandas DataFrame): DataFrame created from the training data CSV file.
    files (list): List of image filenames in the input image directory.
    ss (SearchSpace): Instance of the SearchSpace class for computing the slice and resize
    parameters for object detection.
    """
    def __init__(self, conf=0.25, height_ratio=0.2, width_ratio=0.1, IOS_thresh=0.8,
                 model_type='yolov7', model_path=None, device='cuda:0', config_path=None,
                 outfile='space.json', path='val/', annotations_path='val.json',
                 csv='training_data.csv', custom=False):
        self.conf = conf
        self.height_ratio = height_ratio
        self.width_ratio = width_ratio
        self.IOS_thresh = IOS_thresh
        self.category = ['person', 'bicycle','car', 'motorcycle', 'bus', 'truck', 'vehicle']
        self.model_type = model_type
        self.model_path = model_path
        self.config_path = config_path
        self.path = path
        self.annotations_path = annotations_path
        self.csv = csv
        self.outfile = outfile
        self.data_csv = pd.read_csv(self.csv)
        self.files = [file for file in os.listdir(self.path) if osp.isfile(osp.join(self.path, file))]
        self.files.sort()
        self.ss = SearchSpace()
        self.device = device
        self.custom = custom
    
    def set_categories(self, category_list):
        self.category = category_list
    
    def get_model(self, resize):
        '''
        Returns the object detection model with specified type, confidence threshold and image 
        size.
        '''
        if self.model_type == 'detectron2':
            from sahi.utils.detectron2 import Detectron2TestConstants
            self.model_path = Detectron2TestConstants.FASTERCNN_MODEL_ZOO_NAME
            self.config_path = self.model_path
        return AutoDetectionModel.from_pretrained(
                    model_type=self.model_type,
                    model_path=self.model_path,
                    config_path=self.config_path,
                    confidence_threshold=self.conf,
                    image_size=resize,
                    device=self.device)
    
    def get_predictions(self, resize_width, detection_model, slicee_width, img_path, resize_height,
                        slicee_height):
        result = get_sliced_prediction(
                    self.path+img_path,
                    detection_model,
                    slice_height = self.ss.s_h[slicee_height],
                    slice_width = self.ss.s_w[slicee_width],
                    overlap_height_ratio = self.height_ratio,
                    overlap_width_ratio = self.width_ratio,
                    perform_standard_pred = True,
                    postprocess_type = "GREEDYNMM",
                    postprocess_match_metric = "IOS",
                    postprocess_match_threshold = self.IOS_thresh,
                    auto_slice_resolution = False,
                    resize = True,
                    resize_height = self.ss.r_h[resize_height],
                    resize_width = self.ss.r_w[resize_width])
        return result
    
    def get_annotation_list(self, img_path, result, ann_list):
        data_id = self.data_csv[self.data_csv['Frame_Number']==img_path]    
        ids = data_id['image id'].unique().item()
        print(img_path ,ids)
        ann = result.to_coco_predictions(image_id=ids)
        for i in range(len(ann)):
            if ann[i]['category_name'] in self.category:
                if ann[i]['category_name']=='person':
                    ann[i]['category_id']= 2
                    ann_list.append(ann[i])
                else:
                    ann[i]['category_id'] = 1
                    ann[i]['category_name'] = 'vehicle'
                    ann_list.append(ann[i])
        return ann_list
    
    def save_json(self, slicee, resize, csv_list):
        json_name = str(self.ss.s_w[slicee])+'_'+str(self.ss.r_w[resize])+'_'+self.outfile
        with open(json_name, 'w') as f:
            json.dump(csv_list, f, ensure_ascii=False)
        return json_name
    
    def save_parameters(self, results, resize_width, slicee_width, resize_height, slicee_height):
        results['resize_height'] = self.ss.r_h[resize_height]
        results['resize_width'] = self.ss.r_w[resize_width]
        results['slice_height'] = self.ss.s_h[slicee_height]
        results['slice_width'] = self.ss.s_w[slicee_width]
        results['confidence'] = self.conf
        results['overlap_height_ratio'] = self.height_ratio
        results['overlap_width_ratio'] = self.width_ratio
        results['IOS_thresh'] = self.IOS_thresh
        return results
    
    def save_time(self, elapsed_time_resize, results):
        average_time_image = elapsed_time_resize / len(self.files)
        results['elapsed_time_resize'] = round(elapsed_time_resize,3)
        results['average_time_image'] = round(average_time_image,3)
        return results
    
    def log_time_results(self, elapsed_time_resize, results, resize):
        if resize == 0:
            total_resize_time = elapsed_time_resize * len(self.ss.r_h) * len(self.ss.s_h) * len(self.ss.r_w) * len(self.ss.s_w)
            self.total_time = total_resize_time - elapsed_time_resize
            print('\n\nApproximate total time  = {}\n\n'.format(self.convert_time(self.total_time)))
        else:
            self.total_time -= elapsed_time_resize
            print('\n\nApproximate total time left  = {}\n\n'.format(self.convert_time(self.total_time)))
        print((json.dumps(results, indent=4, sort_keys=False)))
    
    def convert_time(self, seconds):
        if seconds >= 3600:
            hours = seconds // 3600
            return f"{hours} hours"
        elif seconds >= 60:
            minutes = seconds // 60
            return f"{minutes} minutes"
        else:
            return f"{seconds} seconds"

    def evaluate(self):
        """Evaluate the performance of the object detection model.

        This method uses the `get_model` method to retrieve the model, the `get_predictions` method to get the
        object detection results for each image, and the `EvaluationMetrics` class to calculate evaluation metrics.
        The results are then logged in a file "results.txt" and stored as a JSON object.

        Returns:
            None
        """
        with open("results.txt", "a") as filee:
            for slicee_height in range(len(self.ss.s_h)):
                for slicee_width in range(len(self.ss.s_w)):
                    for resize_height in range(len(self.ss.r_h)):
                        for resize_width in range(len(self.ss.r_h)):
                            csv_list =[]
                            ann_list =[]
                            start_time_resize = time.time()
                            print('Running inference for size {}'.format(self.ss.r_h[resize_width]))
                            max_dim = max(self.ss.r_h[resize_height], self.ss.r_w[resize_width])
                            detection_model = self.get_model(max_dim)
                            for img_path in tqdm(self.files):
                                result = self.get_predictions(resize_width, detection_model,
                                    slicee_width, img_path, resize_height, slicee_height)
                                ann_list = self.get_annotation_list(img_path, result, ann_list)
                            csv_list.extend(ann_list)
                            json_name = self.save_json(slicee_width, resize_width, csv_list)
                            if not self.custom:
                                si.modify_json_file(json_name,json_name)
                            if csv_list:
                                evaluation_metrics = EvaluationMetrics(self.annotations_path, json_name, 0.5)
                                results = evaluation_metrics.calculate_metrics()
                                results = self.save_parameters(results, resize_width, slicee_width,
                                    resize_height, slicee_height)
                                end_time_resize = time.time()
                                elapsed_time_resize = end_time_resize - start_time_resize
                                results = self.save_time(elapsed_time_resize, results)
                                self.log_time_results(elapsed_time_resize, results, resize_width)
                                filee.write(json.dumps(results, indent=4, sort_keys=False))
                                filee.flush()
                                new_line = '\n'
                                filee.write(new_line)
                                filee.flush()
                            torch.cuda.empty_cache()

if __name__ == "__main__":   
    parser = argparse.ArgumentParser(description='Explore Class Argparse')
    # add arguments
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--height_ratio', type=float, default=0.2, help='Overlap height ratio')
    parser.add_argument('--width_ratio', type=float, default=0.1, help='Overlap width ratio')
    parser.add_argument('--IOS_thresh', type=float, default=0.8,
        help='Intersection over union threshold')
    parser.add_argument('--model_path', type=str, default='yolov7-e6e.pt', help='path to model')
    parser.add_argument('--model_type', type=str, default='yolov7',
        help='model type : yolov7, detectron2, mmdet')
    parser.add_argument('--device', type=str, default='cuda:0', help='select cpu/gpu')
    parser.add_argument('--cfg_path', type=str, default=None, help='Path to config file')
    parser.add_argument('--path', type=str, default='val/', help='Path to the image folder')
    parser.add_argument('--annotations_path', type=str, default='val.json',
        help='Path to the annotations file')
    parser.add_argument('--csv', type=str, default='training_data.csv', help='Path to the CSV file')
    parser.add_argument('--outfile', type=str, default='space.json', help='Name of the output file')
    parser.add_argument('--custom', action='store_true',
        help='Use this flag if running inference on a custom trained model')
    # create a parser
    args = parser.parse_args()
    # Initializing the class object
    explore_obj = Explore(conf=args.conf, height_ratio=args.height_ratio,
                          width_ratio=args.width_ratio, IOS_thresh=args.IOS_thresh,
                          model_path=args.model_path, model_type=args.model_type,
                          path=args.path, annotations_path=args.annotations_path, csv=args.csv,
                          outfile=args.outfile, custom=args.custom)
    explore_obj.evaluate()

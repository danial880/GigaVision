"""
This script  contains a class  called  "EvaluationMetrics"  which is  used  to
evaluate the performance of object  detection models using the COCO evaluation
metrics. The class takes the path to the COCO annotations file and the path to
the results file, and an  IoU threshold as inputs. It then calculates the true
positive,  false positive,  and false negative counts  and  stores  them  in a
dictionary called `metrics`. The  class  provides  functions  to  convert  the
bounding  box  format,  calculate  the  IoU  between  two  bounding boxes, and
calculate the true positive, false positive, and false negative counts.

Example:
    >>> evaluator = EvaluationMetrics(gt_annotations, dt_annotations)
    >>> metrics = evaluator.calculate_metrics()
    >>> print(metrics)
    {'mAP_all': 0.96,
     'AR_all': 0.92,
     'person_mAP': 0.98,
     'person_AR': 0.95,
     ...
    }
usage: python metric.py --annotations_path /path/to/annotations.json \
                        --results_path /path/to/results.json
"""
import os
import sys
import logging
import argparse
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class EvaluationMetrics:
    """
    Class for evaluating object detection models using the COCO evaluation
    metrics.

    Parameters:
    annotations_path (str): Path to the COCO annotations file.
    results_path (str): Path to the results file.
    iou (float): Intersection over Union (IoU) threshold for evaluation.

    Attributes:
    annotations (dict): Dictionary of COCO annotations loaded from the
                        annotations file.
    results (dict): Dictionary of results loaded from the results file.
    coco_gt (COCO): COCO ground truth object.
    coco_dt (COCO): COCO detections object.
    iou (float): Intersection over Union (IoU) threshold for evaluation.
    iou_type Type (str): Type of IoU computation. Default is 'bbox'.
    metrics (dict): Dictionary to store the evaluation metrics.
    """
    def __init__(self, annotations_path, results_path, iou):
        self.annotations = json.load(open(annotations_path))
        self.results = json.load(open(results_path))
        self.coco_gt = COCO(annotations_path)
        self.coco_dt = self.coco_gt.loadRes(self.results)
        self.iou = iou
        self.iou_type = 'bbox'
        self.metrics = {}

    def convert_bbox_format(self, x_min, y_min, width, height):
        """
        Convert the format of a bounding box from (x_min, ymin, width, height)
        to (x_1, y_1, x_2, y_2).

        Parameters:
        x_min (int): x-coordinate of the top-left corner of the bounding box.
        y_min (int): y-coordinate of the top-left corner of the bounding box.
        width (int): width of the bounding box.
        height (int): height of the bounding box.

        Returns:
        tuple: Tuple of x_1, y_1, x_2, y_2 coordinates of the bounding box.
        """
        x_1 = x_min
        y_1 = y_min
        x_2 = x_min + width
        y_2 = y_min + height
        return x_1, y_1, x_2, y_2

    def calculate_iou(self, ground_truth, prediction):
        """
        Calculates the Intersection over Union (IoU) between two bounding boxes.

        Parameters:
        ground_truth (tuple): A tuple of 4 values representing the top-left and
                              bottom-right coordinates of the ground truth
                              bounding box.
        prediction (tuple): A tuple of 4 values representing the top-left and
                            bottom-right coordinates of the predicted bounding
                            box.

        Returns:
        float: The IoU value, a float in the range [0, 1].
        """
        gt_box_area = (ground_truth[2] - ground_truth[0]) * (ground_truth[3] - ground_truth[1])
        pred_box_area = (prediction[2] - prediction[0]) * (prediction[3] - prediction[1])
        x_1 = max(ground_truth[0], prediction[0])
        y_1 = max(ground_truth[1], prediction[1])
        x_2 = min(ground_truth[2], prediction[2])
        y_2 = min(ground_truth[3], prediction[3])
        intersection = max(x_2 - x_1, 0) * max(y_2 - y_1, 0)
        union = gt_box_area + pred_box_area - intersection
        if union == 0:
            return 0
        return intersection / union

    def __calculate_tp_fp_fn(self):
        """
        Calculate true positives, false positives, and false negatives
        for object detection.

        This function iterates through each ground truth annotation and searches
        for the best prediction with the  highest IoU.If the best prediction has
        an  IoU  greater  than  or equal  to  the  specified  threshold,  it  is
        considered a  true positive.Otherwise,  the ground  truth annotation  is
        considered  a false  negative.Any predictions  not matched  to a  ground
        truth are considered false positives.

        The count of true positives, false positives, and false negatives is
        stored in the metrics dictionary.

        Args:
        annotations: A dictionary containing the ground truth annotations.
        results: A list of predictions.
        iou: The IoU threshold used to determine true positives and false
        negatives.

        Returns:
        None. The count of true positives, false positives, and false negatives
        is stored in the `metrics` dictionary.
        """
        true_positive = 0
        false_positive = 0
        false_negative = 0
        annotations = self.annotations['annotations']
        for annotation in tqdm(annotations, desc='Counting'):
            gt_x, gt_y, gt_width, gt_height = annotation[self.iou_type]
            gt_x1, gt_y1, gt_x2, gt_y2 = self.convert_bbox_format(gt_x, gt_y, gt_width, gt_height)
            max_iou = 0
            best_pred = None
            for result in self.results:
                pred_x, pred_y, pred_width, pred_height = result[self.iou_type]
                pred_x1, pred_y1, pred_x2, pred_y2 = self.convert_bbox_format(
                                                        pred_x, pred_y,
                                                        pred_width, pred_height)
                iou = self.calculate_iou((gt_x1, gt_y1, gt_x2, gt_y2), (pred_x1, pred_y1, pred_x2, pred_y2))
                if iou > max_iou:
                    max_iou = iou
                    best_pred = result
            if max_iou >= self.iou:
                true_positive += 1
                self.results.remove(best_pred)
            else:
                false_negative += 1
        false_positive = len(self.results)
        self.metrics['true_positive'] = true_positive
        self.metrics['false_positive'] = false_positive
        self.metrics['false_negative'] = false_negative

    def coco_evaluate(self, coco_eval):
        """
        Evaluate the results using the COCO evaluation tool.

        Parameters:
        coco_eval (coco evaluation object): Object of the coco evaluation class
        initialized with ground truth and predicted data.

        Returns:
        coco_eval (coco evaluation object): Returns the updated coco evaluation
        object after summarizing the results.
        """
        sys.stdout = open(os.devnull, 'w')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        sys.stdout = sys.__stdout__
        return coco_eval

    def __calculate_map_ar(self):
        """
        Calculates the mean Average Precision (mAP) and Average Recall (AR)
        scores for all categories and each individual category.

        The results are stored in the 'metrics' dictionary attribute with keys
        in the format:
        'mAP_all' for the overall mAP score
        'AR_all' for the overall AR score
        '<category_name>_mAP' for the mAP score of a specific category
        '<category_name>_AR' for the AR score of a specific category

        Args:
            self (COCOMetricCalculator): the instance of the class

        Returns:
            None
        """
        classes = self.coco_gt.getCatIds()
        coco_map_ar = COCOeval(self.coco_gt, self.coco_dt, iouType=self.iou_type)
        coco_map_ar = self.coco_evaluate(coco_map_ar)
        self.metrics['mAP_all'] = round(coco_map_ar.stats[0], 3)
        self.metrics['AR_all'] = round(coco_map_ar.stats[11], 3)
        for clas in classes:
            coco_cw = COCOeval(self.coco_gt, self.coco_dt, iouType=self.iou_type)
            coco_cw.params.catIds = [clas]
            coco_cw = self.coco_evaluate(coco_cw)
            category_name = self.coco_gt.loadCats(clas)[0]['name']
            self.metrics[category_name + '_mAP'] = round(coco_cw.stats[0], 3)
            self.metrics[category_name + '_AR'] = round(coco_cw.stats[11], 3)

    def calculate_metrics(self):
        """
        Calculates  and  returns  the  evaluation  metrics for object detection.
        The metrics include mean average precision (mAP) and average recall (AR)
        for  all classes  and for  each class individually. Also calculates true
        positive  (true_positive),  false positive  (false_positive),  and false
        negative (false_negative) counts.

        Returns:
            dict: A dictionary of evaluation metrics.
        """
        self.__calculate_map_ar()
        self.__calculate_tp_fp_fn()
        return self.metrics


if __name__ == "__main__":
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-ann_pth', '--annotations_path', type=str, help='Path to annotations JSON file')
    parser.add_argument('-rslt_pth', '--results_path', type=str, help='Path to results JSON file')
    parser.add_argument('-iou', '--iou_thresh', default=0.5, type=float, help='IOU Threshold')
    args = parser.parse_args()
    logging.basicConfig(filename='logs.txt', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')
    evaluation_metrics = EvaluationMetrics(args.annotations_path, args.results_path, args.iou_thresh)
    results = evaluation_metrics.calculate_metrics()
    logging.info(json.dumps(results, indent=4, sort_keys=False))
    print(json.dumps(results, indent=4, sort_keys=False))

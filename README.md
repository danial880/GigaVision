# GigaVision

A collection of Jupyter Notebooks for visualizing annotations and generating COCO annotations from GigaVision JSON files.
## VisGiga 

A Jupyter Notebook for visualizing annotations with added features for a better user experience. 

#### Features
- Highlight bounding boxes
- Auto-adjust label font size
- Option to select categories to view

## GigaToCoco
A Jupyter Notebook for generating COCO annotations from JSON files. 

## Getting Started
### Prerequisites
- Jupyter Notebook
- Matplotlib
- NumPy
- PyCOCO Tools

## Explore Search Space
Please install [sahi](https://github.com/danial880/Sahi-Yolov7)
```
python gv/explore.py \
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
```
-  --conf --> Confidence threshold
-  --height_ratio --> Overlap height ratio
-  --width_ratio --> Overlap width ratio
-  --IOS_thresh --> Intersection over union threshold
-  --model_path --> path to model weights
-  --model_type --> type of model (yolov7, detectron2, mmdet)
-  --path --> Path to the image folder
-  --annotations_path --> Path to the annotations file
-  --csv --> Path to the training_data.csv file

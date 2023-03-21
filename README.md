# GigaVision

[GigaVision](https://gigavision.cn) is a program that seeks to revolutionize computer vision when it meets gigapixel videography with both wide field-of-view and high-resolution details. 
## Notebooks
- [VisGiga](https://github.com/danial880/Gigavision/blob/main/notebooks/visualize_detection_dataset/VisGiga.ipynb) 

A Jupyter Notebook for visualizing annotations with added features for a better user experience. 

- [GigaToCoco](https://github.com/danial880/Gigavision/tree/main/notebooks/convert_giga_to_coco) 

A Jupyter Notebook for generating COCO annotations from JSON files. 

- [3D Results Plot](https://github.com/danial880/Gigavision/tree/main/notebooks/3D_plots) 

Plot search space results in 3D. 
- [Analyze Tracking Sequences](https://github.com/danial880/Gigavision/tree/main/notebooks/analyze_tracking_dataset) 

Visualize bounding box distribution of each tracking ID in a particular sequence.

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

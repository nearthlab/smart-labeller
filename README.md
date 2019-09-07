# smart-labeller
Labelling tool for creating custom datasets for object detection and semantic / instance segmentation.
Intended to make labelling segmentation mask easier for abstract objects (e.g. cracks or erosion on surfaces) using algorithms such as  [GrabCut](https://docs.opencv.org/3.4/d8/d83/tutorial_py_grabcut.html) and image thresholding.

![demo](defect_labeller_demo.gif)
<br/>Labelling defects on wind turbines' blade images using this repository

# Installation
``` bash
git clone https://github.com/nearthlab/smart-labeller
cd smart-labeller
pip install -r requirements.txt
```

# Usage
First, prepare images and class_labels.json as in the [datasets/sample](https://github.com/nearthlab/smart-labeller/tree/master/datasets/sample).
1. Create labels for new images or edit existing labels
``` bash
python label.py [(optional) /path/to/dataset]
# press F1 to see instructions
```

2. Augment existing labels
``` bash
python augment.py [(optional) /path/to/dataset]
# set options as you wish and press augment button
```

3. Export your labels into one of PASCAL-VOC / COCO / CityScapes (a.k.a. KITTI) dataset.
``` bash
python export.py [(optional) /path/to/dataset]
# set options as you wish and press export button
```

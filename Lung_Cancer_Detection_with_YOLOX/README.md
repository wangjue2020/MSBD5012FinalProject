# Lung Cancer Detection with YOLOX

* To run YOLOX, you should prepare the environment follow the guidance from  https://github.com/Megvii-BaseDetection/YOLOX
* To get the data set, you can open  this link https://drive.google.com/drive/folders/13Tktp7AVmkIJsshdGBGdAM-8IEJVbvZM on browser and download the zip file named "cancer", and unzip the files with in the cancer directory into YOLOX/datasets/cancer
* The history results we have gotten from YOLOX is saved as txt files in YOLOX/results. You can open the FindBestResult.ipynb by jupyter notebook to get the best result from the four txt files.
* The history weights (checkpoints) are saved in YOLOX/weights/cancer_detector
* This project is an experiment based on yolox2021 by integrating datasets to the one fits to yolox. 
The detail for yolox, please refers to the https://github.com/Megvii-BaseDetection/YOLOX

files modified for this project are:
[cancer.py](./Lung_Cancer_Detection_with_YOLOX/YOLOX/exps/cancer.py)
[coco_classes.py](./Lung_Cancer_Detection_with_YOLOX/YOLOX/yolox/data/datasets/coco_classes.py)

Using our own data augmentations and preprocessing tools
[data augmentations](./data_preprocess)

If you use YOLOX in your research, please cite our work by using the following BibTeX entry:

```latex
 @article{yolox2021,
  title={YOLOX: Exceeding YOLO Series in 2021},
  author={Ge, Zheng and Liu, Songtao and Wang, Feng and Li, Zeming and Sun, Jian},
  journal={arXiv preprint arXiv:2107.08430},
  year={2021}
}
```

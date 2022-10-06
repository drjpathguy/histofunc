# HistoFunc
======


QuPath
======

**QuPath is open source software for bioimage analysis**.



Functions to facilitate the development of python algorithms for histology and digital pathology.

Features include:

* Tool that creates patches out of a whole slide image
- Patch size, pixel overlap, padding, and the ability to skip patches can be specified. 
- 3 methods are included. One method packages the images into a single array [N x W x H x C] where N is the number of patches. A second method created a dictionairy that contains all the patches as individual [W x H x C] images. The final method stores the location of each patch on file, and uses a second function to pull the patches. This is done without initially loading the WSI, which can be advantageous when used in parallel processing and when memory storage needs to be minimized.
* Color deconvolution applied through numpy and using PyTorch. 

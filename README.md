# HAI-SFM (Historical Aerial Image - Structure-from-Motion)

## Introduction

HAI-SFM provides code to the publication XXX.
It serves as a tool for an end-to-end Structure-from-Motion pipeline for historical aerial images.
It modifies the feature matching methods [DISK](https://github.com/cvlab-epfl/disk) and [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork) to be usable on images with high resolution.
Therefore, the default workflow downsamples the original images, finds corresponding image pairs and uses a modified parameter setting to match the original sized images.
The derived feature matches are directly imported into a [COLMAP](https://github.com/colmap/colmap) database file so that the scene can be reconstructed and the camera parameters are estimated.
We provide a mono-temporal and multi-temporal image sample for initial testing.

If you are using this project for your research, please cite:
```
@article{Maiwald_HAI-SFM_2023,
    author = {Ferdinand Maiwald and Denis Feurer and Anette Eltner},
    title = {Solving photogrammetric cold cases using AI-based image matching: New potential for monitoring the past with historical aerial images},
    journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
    volume = {206},
    pages = {184-200},
    year = {2023},
    issn = {0924-2716},
    doi = {https://doi.org/10.1016/j.isprsjprs.2023.11.008},
    url = {https://www.sciencedirect.com/science/article/pii/S0924271623003131},
    keywords = {Historical aerial images, Feature matching, Neural networks, Structure-from-motion, Digital surface model, Multi-temporal},
```

If you are using the sample dataset for your research, please cite:
```
@article{isprs-archives-XLIII-B2-2022-1175-2022,
    AUTHOR = {Farella, E. M. and Morelli, L. and Remondino, F. and Mills, J. P. and Haala, N. and Crompvoets, J.},
    TITLE = {THE EUROSDR TIME BENCHMARK FOR HISTORICAL AERIAL IMAGES},
    JOURNAL = {The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences},
    VOLUME = {XLIII-B2-2022},
    YEAR = {2022},
    PAGES = {1175--1182},
    URL = {https://isprs-archives.copernicus.org/articles/XLIII-B2-2022/1175/2022/},
    DOI = {10.5194/isprs-archives-XLIII-B2-2022-1175-2022}
}
```
We use only a small part of the original images and downsampled the images to half of the original resolution.
The images are part of the [TIME](https://time.fbk.eu) dataset realized with the support of EuroSDR and different European Mapping Agencies. 
In particular, the reported images are part of the Normay dataset, kindly provided by Mikko Sippo. 
For more information and the full resolution images, please check the TIME [website](https://time.fbk.eu).

## Installation and dependencies

For feature detection and feature matching, we recommend using a GPU.
However, it should also work (significantly slower) on CPU.

Steps for installation are the following:

1. Clone the repository
2. The dependencies are provided in the `requirements.txt` and in the respective feature matchers
3. Most of the dependencies rely on Python 3.7, but we ran successful tests on 3.8 and 3.10
4. Move to the directory `cd HAI-SFM` and run `pip install -r requirements.txt`

## Getting started

The default function call with all required arguments is:   
`python main.py --image_dir data/multi-temporal --superglue_path SuperGluePretrainedNetwork --disk_path disk`  

### Parameters
The call can be extended using the following arguments:
- `--gpu`: Enable GPU usage (recommended!)
- `--gpu_device`: Specify the GPU device index
- `--image_dir`: Path to original images (currently .jpg, .png, or .tif are supported)
- `--config`: Choice between default, tile-based approach, and disk approach as explained in the publication
- `--rotation`: Specification if images are rotated correctly in between flight strips (default is 'not-rotated')
- `--flightstrips`: Number of flightstrips if known. If the rotation is unknown this specifices the number of iteration to find the correct rotation (default = 10)
- `--colmap`: Generate a COLMAP database file from all types of feature matches (useful when continuing with SfM)

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
@article{Maiwald,
    author={Maiwald, Ferdinand and Feurer, Denis and Eltner, Anette}
    title=XXX
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

### Running on sample image pair


## Known issues

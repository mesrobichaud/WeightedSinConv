# WeightedSinConv
An adaptation of Harmonic Embeddings with support for more pixel-wise weight options and guide functions.

# Harmonic Embeddings 

The harmonic directory is adapted from [Kulikov and Lempitsky](https://github.com/kulikovv/harmonic) with the following modifications: 
- Support for harmonic square wave approximations (sq_embeddings.py, sq_unet.py, sinconv/SquareConv)
- Easier pre-trained backbone integration and multiple variants of unet in unet.py

The Loader file functionalizes methods provided in train_CVPPP from the original project and adds more functionality:
- pixel-wise weighted loss (loader/make_weight_map)
- specialized file readers for CVPPP (RGB) and HeLa (greyscale) image sets 
- evaluation metrics (BDC, IOU, DiC)

## Navigation 
1. hela_trainer notebook is used to train the network with or without pre-training. 
2. Open demo or testing notebooks to evaluate the model

## Results 

Results from the test HeLa set for the two top networks (edge width of 5, and distance-based method, wx) are shown along with the ground truth for three images in the **img** directory.

## Selecting Weight Method 

Choose from add_edges or add_borders (uses make_weight_map) to generate the weight map from the labelled images. 

## K-Means for Instance Segmentation

The classical.ipynb notebook demonstrates a k-means implementation to compare to the deep learning method. Images of its performance are saved to **img/km**. 
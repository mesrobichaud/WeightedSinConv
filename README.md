# WeightedSinConv
An adaptation of Harmonic Embeddings. 

# Harmonic Embeddings 

The harmonic directory is adapted from [Kulikov and Lempitsky](https://github.com/kulikovv/harmonic) with the following modifications: 
- Added pixel-wise weighted loss (loader/make_weight_map)
- Support for harmonic square wave approximations (sq_embeddings.py, sq_unet.py, sinconv/SquareConv)
- Easier pre-trained backbone integration and multiple variants of unet in unet.py

## Selecting Weight Method 

Choose from  

### Contributions 
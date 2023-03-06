import torch

class GraphBatchTransform:
    """
    Class that takes in an image and transforms it into a sequence a Transformer can process. 
    If patch is greater than 0, it will separate the image into patches and apply a random linear projection to output dim
    """
    def __init__(input_dim, patch_size=None, output_dim=8):
        self.input_dim = input_dim
        self.patch_size = patch_size
        self.output_dim = output_dim

        if self.patch>0:
            num_patches = (self.input_dim[1] // patch_size) * (self.input_dim[2] // patch_size)
            patch_dim =  self.input_dim[0] * patch_size * patch_size

        
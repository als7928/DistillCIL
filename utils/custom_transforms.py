import torch

############################################
class PermuteTensor:
    # Tansform Module for Permutation Setting
    def __init__(self, input_shape):
        self.input_shape = input_shape
        c, h, w = input_shape
        self.perm = torch.randperm(c*h*w)

    def __call__(self, img):
        img = img.reshape(-1)
        return img[self.perm].reshape(*self.input_shape)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
############################################
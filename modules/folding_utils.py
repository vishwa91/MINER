import torch

class UnfoldNd(torch.nn.Module):
    '''
        Unfolding operator for 3D tensors with disjoint folding. The operator
        is a drop in replacement for unfoldNd.UnfoldNd
        
        Inputs:
            kernel_size: Folding kernel size
            stride: Folding stride. This is just for compatibility, as the 
                stride is fixed to be the same as kernel_size
            tensor: (N, C, H, W, T) sized tensor
                
        Outputs:
            unfolded_tensor: (N, kernel_size**3, -1) sized unfolded tensor
    '''
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        
        self.ksize = kernel_size
        self.stride = stride

    def forward(self, tensor):
        N, C, H, W, T = tensor.shape
        fold1 = tensor.unfold(2, self.ksize, self.ksize)
        fold2 = fold1.unfold(3, self.ksize, self.ksize)
        fold3 = fold2.unfold(4, self.ksize, self.ksize)
        
        collapse1 = fold3.reshape(N, -1, self.ksize, self.ksize, self.ksize)
        collapse2 = collapse1.reshape(N, -1, self.ksize**3)
        
        # To maintain compatibility, swap the two axes
        collapse3 = collapse2.permute(0, 2, 1)
        
        return collapse3
    
class FoldNd(torch.nn.Module):
    '''
        Folding operator for 3D tensors with disjoint folding. The operator
        is a drop in replacement for unfoldNd.FoldNd
        
        Inputs:
            output_size: (H, W, T) -- Three-tuple with output size
            kernel_size: Folding kernel size
            stride: Folding stride. This is just for compatibility, as the 
                stride is fixed to be the same as kernel_size
                
        Outputs:
            folded_tensor: (N, C, H, W, T) sized folded tensor
    '''
    def __init__(self, output_size, kernel_size, stride=None):
        super().__init__()
        
        self.output_size = output_size
        self.ksize = kernel_size
        self.stride = stride
        
        # Create a folded set of indices here
        self.numel = output_size[0]*output_size[1]*output_size[2]
        idx = torch.arange(self.numel, dtype=torch.int64)
        idx_cube = idx.reshape(1, 1, *output_size)
        
        self.unfolding = UnfoldNd(self.ksize)
        self.folded_idx = self.unfolding(idx_cube)

    def forward(self, tensor, output=None):
        if output is None:
            output = torch.zeros(self.numel, device=tensor.device)
        else:
            output = output.flatten()
            
        output[self.folded_idx] = tensor
        output_cube = output.reshape(1, 1, *self.output_size)
        
        return output_cube


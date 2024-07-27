import torch
from torch.utils.cpp_extension import load_inline
from .pair_wise_distance_cuda_source import source


print("compile cuda source of 'pair_wise_distance' function...")
print("NOTE: if you avoid this process, you make .cu file and compile it following https://pytorch.org/tutorials/advanced/cpp_extension.html")
# pair_wise_distance_cuda = load_inline(
#     "pair_wise_distance", cpp_sources="", cuda_sources=source
# )
print("done")


# class PairwiseDistFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(self, pixel_features, spixel_features, init_spixel_indices, num_spixels_width, num_spixels_height):
#         self.num_spixels_width = num_spixels_width
#         self.num_spixels_height = num_spixels_height
#         output = pixel_features.new(pixel_features.shape[0], 9, pixel_features.shape[-1]).zero_()
#         self.save_for_backward(pixel_features, spixel_features, init_spixel_indices)

#         return pair_wise_distance_cuda.forward(
#             pixel_features.contiguous(), spixel_features.contiguous(),
#             init_spixel_indices.contiguous(), output,
#             self.num_spixels_width, self.num_spixels_height)

#     @staticmethod
#     def backward(self, dist_matrix_grad):
#         pixel_features, spixel_features, init_spixel_indices = self.saved_tensors

#         pixel_features_grad = torch.zeros_like(pixel_features)
#         spixel_features_grad = torch.zeros_like(spixel_features)
        
#         pixel_features_grad, spixel_features_grad = pair_wise_distance_cuda.backward(
#             dist_matrix_grad.contiguous(), pixel_features.contiguous(),
#             spixel_features.contiguous(), init_spixel_indices.contiguous(),
#             pixel_features_grad, spixel_features_grad,
#             self.num_spixels_width, self.num_spixels_height
#         )
#         return pixel_features_grad, spixel_features_grad, None, None, None
    

class PairwiseDistFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pixel_features, spixel_features, init_spixel_indices, num_spixels_width, num_spixels_height):
        ctx.num_spixels_width = num_spixels_width
        ctx.num_spixels_height = num_spixels_height
        ctx.save_for_backward(pixel_features, spixel_features, init_spixel_indices)

        batch_size, num_features = pixel_features.shape[0], pixel_features.shape[-1]
        output = pixel_features.new_zeros(batch_size, 9, num_features)

        # Implementing the forward pass using PyTorch operations
        for b in range(batch_size):
            for i in range(num_spixels_height):
                for j in range(num_spixels_width):
                    spixel_index = i * num_spixels_width + j
                    spixel_feat = spixel_features[b, spixel_index]
                    pixel_feat = pixel_features[b, init_spixel_indices[b, i, j]]
                    output[b, spixel_index % 9] = (pixel_feat - spixel_feat).pow(2).sum(-1).sqrt()

        return output

    @staticmethod
    def backward(ctx, dist_matrix_grad):
        pixel_features, spixel_features, init_spixel_indices = ctx.saved_tensors
        num_spixels_width, num_spixels_height = ctx.num_spixels_width, ctx.num_spixels_height

        pixel_features_grad = torch.zeros_like(pixel_features)
        spixel_features_grad = torch.zeros_like(spixel_features)

        # Implementing the backward pass using PyTorch operations
        for b in range(pixel_features.shape[0]):
            for i in range(num_spixels_height):
                for j in range(num_spixels_width):
                    spixel_index = i * num_spixels_width + j
                    spixel_feat = spixel_features[b, spixel_index]
                    pixel_feat = pixel_features[b, init_spixel_indices[b, i, j]]
                    grad = dist_matrix_grad[b, spixel_index % 9]

                    pixel_diff = pixel_feat - spixel_feat
                    pixel_grad = grad * pixel_diff / (pixel_diff.pow(2).sum(-1).sqrt() + 1e-10)
                    spixel_grad = -pixel_grad

                    pixel_features_grad[b, init_spixel_indices[b, i, j]] += pixel_grad
                    spixel_features_grad[b, spixel_index] += spixel_grad

        return pixel_features_grad, spixel_features_grad, None, None, None
    

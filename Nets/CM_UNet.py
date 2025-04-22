import os                    # Operating system interface for file and directory operations
import time                  # Functions related to time, such as timing and getting current time
import math                  # Mathematical functions, such as trigonometric functions, logarithms, exponentiation, etc.
import copy                  # Provides shallow and deep copy operations
from functools import partial # Higher-order function tool that allows fixing some parameters or keyword arguments of a function
from typing import Optional, Callable, Any  # Type annotation tools, Optional for optional types, Callable for function types, Any for any type
from collections import OrderedDict         # Ordered dictionary, stores key-value pairs in insertion order

import torch                 # Main PyTorch package for tensor computation and automatic differentiation
import torch.nn as nn        # PyTorch's neural network module, containing various neural network layers and loss functions
import torch.nn.functional as F  # Functional interface for defining neural network layers in PyTorch
import torch.utils.checkpoint as checkpoint  # Checkpointing mechanism for memory optimization, storing and recomputing to save memory
from einops import rearrange, repeat # Powerful tensor manipulation library, rearrange for reordering tensor dimensions, repeat for repeating tensors along specific dimensions
from timm.models.layers import DropPath, trunc_normal_  # Model layers from timm library, DropPath for random depth implementation, trunc_normal_ for truncated normal distribution initialization
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count  # Used for calculating model FLOPs (Floating Point Operations) and parameter count

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})" # Allows seeing the dropout probability of DropPath objects, helpful for understanding and debugging the code


# import selective scan ============================== Custom CUDA extension module designed to accelerate and optimize selective scan operations
try:
    import selective_scan_cuda_oflex
except Exception as e:
    ...
    # print(f"WARNING: can not import selective_scan_cuda_oflex.", flush=True)
    # print(e, flush=True)

try:
    import selective_scan_cuda_core
except Exception as e:
    ...
    # print(f"WARNING: can not import selective_scan_cuda_core.", flush=True)
    # print(e, flush=True)

try:
    import selective_scan_cuda
except Exception as e:
    ...
    # print(f"WARNING: can not import selective_scan_cuda.", flush=True)
    # print(e, flush=True)


# fvcore flops ======================================= Used for calculating the FLOPs (Floating Point Operations) of selective scan operations
def flops_selective_scan_fn(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_complex=False):
    """
    B: Batch size, default is 1
    L: Sequence length, default is 256
    D: Feature dimension, default is 768
    N: Size of a specific dimension, default is 16
    with_D: Boolean, indicates whether to include D dimension calculations, default is True
    with_Z: Boolean, indicates whether to include Z dimension calculations, default is False
    with_complex: Boolean, indicates whether to include complex calculations, default is False

    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu]
    """
    assert not with_complex
    # https://github.com/state-spaces/mamba/issues/110
    flops = 9 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    return flops

# Reference function to calculate FLOPs for selective scan operations
def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu]
    """
    import numpy as np

    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop


    assert not with_complex

    flops = 0 # below code flops = 0

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")

    in_for_flops = B * D * N
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    return flops

# Print debug names of input tensors for easier tracking and understanding of data flow
def print_jit_input_names(inputs):
    print("input params: ", end=" ", flush=True)
    try:
        for i in range(10):
            print(inputs[i].debugName(), end=" ", flush=True)
    except Exception as e:
        pass
    print("", flush=True)



# cross selective scan ===============================
# Custom CUDA extension module for selective scan operation acceleration, providing efficient forward and backward propagation calculations
class SelectiveScanMamba(torch.autograd.Function):
    # comment all checks if inside cross_selective_scan
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1, oflex=True):
        # assert nrows in [1, 2, 3, 4], f"{nrows}" # 8+ is too slow to compile
        # assert u.shape[1] % (B.shape[1] * nrows) == 0, f"{nrows}, {u.shape}, {B.shape}"
        ctx.delta_softplus = delta_softplus
        # all in float
        # if u.stride(-1) != 1:
        #     u = u.contiguous()
        # if delta.stride(-1) != 1:
        #     delta = delta.contiguous()
        # if D is not None and D.stride(-1) != 1:
        #     D = D.contiguous()
        # if B.stride(-1) != 1:
        #     B = B.contiguous()
        # if C.stride(-1) != 1:
        #     C = C.contiguous()
        # if B.dim() == 3:
        #     B = B.unsqueeze(dim=1)
        #     ctx.squeeze_B = True
        # if C.dim() == 3:
        #     C = C.unsqueeze(dim=1)
        #     ctx.squeeze_C = True

        out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, delta_bias, delta_softplus)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()

        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, None, delta_bias, dout, x, None, None, ctx.delta_softplus,
            False
        )
        # dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        # dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)

# A different one using the selective_scan_cuda_core module
class SelectiveScanCore(torch.autograd.Function):
    # comment all checks if inside cross_selective_scan
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1, oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda_core.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_core.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)

# A different one using the selective_scan_cuda_oflex module
class SelectiveScanOflex(torch.autograd.Function):
    # comment all checks if inside cross_selective_scan
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1, oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda_oflex.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1, oflex)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_oflex.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)

# Used to simulate the forward and backward propagation of the selective scan operation. Its main role is as a placeholder or debugging tool, without performing actual computations
class SelectiveScanFake(torch.autograd.Function):
    # comment all checks if inside cross_selective_scan
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1, oflex=True):
        ctx.delta_softplus = delta_softplus
        ctx.backnrows = backnrows
        x = delta
        out = u
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias = u * 0, delta * 0, A * 0, B * 0, C * 0, C * 0, (D * 0 if D else None), (delta_bias * 0 if delta_bias else None)
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


# =============

# Extract all elements on the anti-diagonal of a tensor and concatenate them into a new tensor
def antidiagonal_gather(tensor):
    # Extract all anti-diagonal elements and concatenate them (from top-right to bottom-left diagonal)
    B, C, H, W = tensor.size()
    shift = torch.arange(H, device=tensor.device).unsqueeze(1)  # Create a column vector [H, 1]
    index = (torch.arange(W, device=tensor.device) - shift) % W  # Create an index matrix [H, W] using broadcasting
    # Expand the index to match B and C dimensions
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    # Use gather to select elements based on the index
    return tensor.gather(3, expanded_index).transpose(-1,-2).reshape(B, C, H*W)

def diagonal_gather(tensor):
    # Extract all diagonal elements and concatenate them (from top-left to bottom-right diagonal)
    B, C, H, W = tensor.size()
    shift = torch.arange(H, device=tensor.device).unsqueeze(1)  # Create a column vector [H, 1]
    index = (shift + torch.arange(W, device=tensor.device)) % W  # Create an index matrix [H, W] using broadcasting
    # Expand the index to match B and C dimensions
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    # Use gather to select elements based on the index
    return tensor.gather(3, expanded_index).transpose(-1,-2).reshape(B, C, H*W)

def diagonal_scatter(tensor_flat, original_shape):
    # Restore the diagonal elements concatenated into a 1D vector back to their original matrix form (from top-left to bottom-right diagonal)
    B, C, H, W = original_shape
    shift = torch.arange(H, device=tensor_flat.device).unsqueeze(1)  # Create a column vector [H, 1]
    index = (shift + torch.arange(W, device=tensor_flat.device)) % W  # Create an index matrix [H, W] using broadcasting
    # Expand the index to match B and C dimensions
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    # Create an empty tensor to store the result of the scatter operation
    result_tensor = torch.zeros(B, C, H, W, device=tensor_flat.device, dtype=tensor_flat.dtype)
    # Reshape the flattened tensor to [B, C, W, H], considering the need to use transpose to swap H and W
    tensor_reshaped = tensor_flat.reshape(B, C, W, H).transpose(-1, -2)
    # Use scatter_ to place the elements back to their original positions based on the index
    result_tensor.scatter_(3, expanded_index, tensor_reshaped)
    return result_tensor

def antidiagonal_scatter(tensor_flat, original_shape):
    # Restore the anti-diagonal elements concatenated into a 1D vector back to their original matrix form (from top-right to bottom-left diagonal)
    B, C, H, W = original_shape
    shift = torch.arange(H, device=tensor_flat.device).unsqueeze(1)  # Create a column vector [H, 1]
    index = (torch.arange(W, device=tensor_flat.device) - shift) % W  # Create an index matrix [H, W] using broadcasting
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    # Initialize a tensor with the same shape as the original tensor, filled with zeros
    result_tensor = torch.zeros(B, C, H, W, device=tensor_flat.device, dtype=tensor_flat.dtype)
    # Reshape the flattened tensor to [B, C, W, H], because the operation is collecting along the last dimension, we need to adjust the shape and swap the dimensions
    tensor_reshaped = tensor_flat.reshape(B, C, W, H).transpose(-1, -2)
    # Use scatter_ to place the elements back to their original positions based on the index
    result_tensor.scatter_(3, expanded_index, tensor_reshaped)
    return result_tensor

class CrossScan(torch.autograd.Function):
    # Flatten the image in a specific direction, remove diagonal and anti-diagonal operations, retain horizontal and vertical scans
    @staticmethod
    def forward(ctx, x: torch.Tensor): # Forward pass, splitting into four directions
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        # Create an empty tensor to store the horizontal and vertical scan results
        xs = x.new_empty((B, 4, C, H * W))
        # Add horizontal and vertical scans
        xs[:, 0] = x.flatten(2, 3)  # Horizontal scan
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)  # Vertical scan
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1]) # Reverse scan for the above two steps

        return xs # Matrix containing scan results from different directions

    @staticmethod
    def backward(ctx, ys: torch.Tensor): # Restore the flattened and concatenated tensor ys back to its original matrix form
        B, C, H, W = ctx.shape
        L = H * W
        # Reverse the vertical and horizontal parts and add them to the original horizontal and vertical scans
        y_rb = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        # Convert vertical part to horizontal, then add them, and restore to the original matrix form
        y_rb = y_rb[:, 0] + y_rb[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        y_rb = y_rb.view(B, -1, H, W)

        return y_rb

class CrossMerge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor): # Reassemble the output from CrossScan back into the original shape of the tensor
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        # Reverse the vertical and horizontal parts and add them to the original horizontal and vertical scans
        y_rb = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        # Convert vertical part to horizontal, then add them, and restore to the original matrix form
        y_rb = y_rb[:, 0] + y_rb[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        y_rb = y_rb.view(B, -1, H, W)

        return y_rb.view(B, D, -1)

    @staticmethod
    def backward(ctx, x: torch.Tensor): # Flatten and concatenate the restored tensor for gradient calculation
        H, W = ctx.shape
        B, C, L = x.shape
        # Create an empty tensor to store the horizontal and vertical scan results
        xs = x.new_empty((B, 4, C, L))

        # Horizontal and vertical scans
        xs[:, 0] = x
        xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])

        return xs.view(B, 4, C, H, W)

# these are for ablations =============(消融实验)
# CrossScan_Ab_2direction class implements a simplified version of tensor flattening and restoration, involving only two directions.
# Compared to the CrossScan class, CrossScan_Ab_2direction only flattens the input tensor horizontally and restores these flattened results to their original shape during the backward pass.
# This simplified operation can be used for specific tests or ablation experiments to evaluate the impact of different flattening directions on model performance.

# Flatten and flip the input tensor during the forward pass, restore it during the backward pass
class CrossScan_Ab_2direction(torch.autograd.Function): # Focuses on flattening and flipping the input tensor
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 4, C, H * W))
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        L = H * W
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        return y.view(B, -1, H, W)


class CrossMerge_Ab_2direction(torch.autograd.Function): # Focuses on merging tensors that are flattened in multiple directions
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        y = ys.sum(dim=1)
        return y

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 4, C, L))
        xs[:, 0] = x
        xs[:, 1] = x
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        xs = xs.view(B, 4, C, H, W)
        return xs


class CrossScan_Ab_1direction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.view(B, 1, C, H * W).repeat(1, 4, 1, 1).contiguous()
        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        y = ys.sum(dim=1).view(B, C, H, W)
        return y


class CrossMerge_Ab_1direction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        y = ys.sum(dim=1).view(B, D, H * W)
        return y

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.view(B, 1, C, L).repeat(1, 4, 1, 1).contiguous().view(B, 4, C, H, W)
        return xs

# CrossScan_Ab_2direction and CrossMerge_Ab_2direction involve flip and flatten operations. These are suitable for situations that require flattening and merging in multiple directions.
# CrossScan_Ab_1direction and CrossMerge_Ab_1direction only involve simple flattening and repetition operations. These are more suitable for cases where simple flattening and repetition are needed.


# =============
# ZSJ Here is the specific content for mamba. To add scanning directions, modify it here.
def cross_selective_scan(
    x: torch.Tensor=None, # Input tensor with shape (B, D, H, W), where B is batch size, D is number of channels, H is height, W is width
    x_proj_weight: torch.Tensor=None, # Weight tensor for projecting the input tensor, shape (K, C, D), where K represents the number of scanning directions
    x_proj_bias: torch.Tensor=None, # Bias tensor for projecting the input tensor, shape (K, C)
    dt_projs_weight: torch.Tensor=None, # Weight tensor for projecting dts tensor, shape (K, D, R), R is the dimension after projection
    dt_projs_bias: torch.Tensor=None, # Bias tensor for projecting dts tensor, shape (K, D)
    A_logs: torch.Tensor=None, # Log form of matrix A parameters, shape (D, N), used for selective scan
    Ds: torch.Tensor=None, # Parameters of vector D, shape (K * C), used for selective scan
    delta_softplus = True,  # Boolean value indicating whether to apply the softplus activation function in the selective scan
    out_norm: torch.nn.Module=None, # A normalization module to apply to the output tensor. It could be LayerNorm, Softmax, Sigmoid, etc.
    out_norm_shape="v0", # A string indicating the shape of the output tensor after normalization. Defaults to "v0". Possible values include "v0" and "v1"
    # ==============================
    to_dtype=True, # A boolean indicating whether to convert the output tensor to the input tensor's dtype
    force_fp32=False, # A boolean indicating whether to force the tensor to be converted to float32
    # ==============================
    nrows = -1, # Parameter for SelectiveScanNRow, indicating the number of rows for selective scan. 0 means auto-select, -1 means disabled. Defaults to -1
    backnrows = -1, # Indicates the number of rows for reverse selective scan. 0 means auto-select, -1 means disabled. Defaults to -1
    ssoflex=True, # A boolean indicating whether to output float32 in SSOflex. If False, SSOflex behaves like SSCore
    # ==============================
    SelectiveScan=None, # A custom selective scan function
    CrossScan=CrossScan, # A custom cross scan function
    CrossMerge=CrossMerge, # A custom cross merge function
):
    # out_norm: whatever fits (B, L, C); LayerNorm; Sigmoid; Softmax(dim=1);...

    B, D, H, W = x.shape
    D, N = A_logs.shape
    K, D, R = dt_projs_weight.shape
    L = H * W # Flattened length

    # Dynamically set the value of nrows based on the number of channels D
    if nrows == 0:
        if D % 4 == 0:
            nrows = 4
        elif D % 3 == 0:
            nrows = 3
        elif D % 2 == 0:
            nrows = 2
        else:
            nrows = 1

    if backnrows == 0:
        if D % 4 == 0:
            backnrows = 4
        elif D % 3 == 0:
            backnrows = 3
        elif D % 2 == 0:
            backnrows = 2
        else:
            backnrows = 1

    def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
        return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, backnrows, ssoflex)

    xs = CrossScan.apply(x) # Apply CrossScan class's apply method to the input tensor x, performing specific scan operations and returning the result xs

    # Use einsum operation to perform multi-dimensional multiplication and summation between input tensor xs and weight tensor x_proj_weight
    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
    if x_proj_bias is not None:
        x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)
    xs = xs.view(B, -1, L)
    dts = dts.contiguous().view(B, -1, L)
    As = -torch.exp(A_logs.to(torch.float)) # (k * c, d_state)
    Bs = Bs.contiguous()
    Cs = Cs.contiguous()
    Ds = Ds.to(torch.float) # (K * c)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)

    if force_fp32:
        xs = xs.to(torch.float)
        dts = dts.to(torch.float)
        Bs = Bs.to(torch.float)
        Cs = Cs.to(torch.float)
    # ZSJ Here, the matrix is split into different direction sequences, and scanning is performed
    ys: torch.Tensor = selective_scan(
        xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
    ).view(B, K, -1, H, W)
    # ZSJ Here, the processed sequences are merged and restored back to the original matrix form
    y: torch.Tensor = CrossMerge.apply(ys)

    if out_norm_shape in ["v1"]: # (B, C, H, W)
        y = out_norm(y.view(B, -1, H, W)).permute(0, 2, 3, 1) # (B, H, W, C)
    else: # (B, L, C)
        y = y.transpose(dim0=1, dim1=2).contiguous() # (B, L, C)
        y = out_norm(y).view(B, H, W, -1)

    return (y.to(x.dtype) if to_dtype else y)


# Used to calculate the floating point operations (FLOPs) for the selective scan operation
def selective_scan_flop_jit(inputs, outputs):
    print_jit_input_names(inputs)
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_selective_scan_fn(B=B, L=L, D=D, N=N, with_D=True, with_Z=False)
    return flops

#=====================================================
class ResConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Add a 1x1 convolution layer for channel matching
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.bn_residual = nn.BatchNorm2d(out_channels)


    def forward(self, x):
        # Main path
        residual = self.residual(x)
        residual = self.bn_residual(residual)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # Add the residual connection to the output
        out += residual
        out = self.relu(out)
        return out

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # Bilinear interpolation
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # Convolution
            nn.BatchNorm2d(out_channels),  # Batch normalization
            nn.ReLU(inplace=True)  # Activation function
        )

    def forward(self, x):
        return self.up(x)

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x


class OSSM(nn.Module):
    def __init__(
        self,
        d_model=768,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        d_conv=3,
        conv_bias=True,
        dropout=0.0,
        bias=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v0",
        forward_type="v2",
        **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_conv = d_conv

        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        self.disable_force32, forward_type = checkpostfix("no32", forward_type)
        self.disable_z, forward_type = checkpostfix("noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("nozact", forward_type)

        if forward_type[-len("none"):] == "none":
            forward_type = forward_type[:-len("none")]
            self.out_norm = nn.Identity()
        elif forward_type[-len("dwconv3"):] == "dwconv3":
            forward_type = forward_type[:-len("dwconv3")]
            self.out_norm = nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False)
            self.out_norm_shape = "v1"
        elif forward_type[-len("softmax"):] == "softmax":
            forward_type = forward_type[:-len("softmax")]
            self.out_norm = nn.Softmax(dim=1)
        elif forward_type[-len("sigmoid"):] == "sigmoid":
            forward_type = forward_type[:-len("sigmoid")]
            self.out_norm = nn.Sigmoid()
        else:
            self.out_norm = nn.LayerNorm(d_inner)

        FORWARD_TYPES = dict(
            v0=self.forward_corev0,
            v2=partial(self.forward_corev2, force_fp32=True, SelectiveScan=SelectiveScanCore),
            v3=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex),
            v31d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, cross_selective_scan=partial(
                cross_selective_scan, CrossScan=CrossScan_Ab_1direction, CrossMerge=CrossMerge_Ab_1direction,
            )),
            v32d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, cross_selective_scan=partial(
                cross_selective_scan, CrossScan=CrossScan_Ab_2direction, CrossMerge=CrossMerge_Ab_2direction,
            )),
            fake=partial(self.forward_corev2, force_fp32=True, SelectiveScan=SelectiveScanFake),
            v1=partial(self.forward_corev2, force_fp32=True, SelectiveScan=SelectiveScanOflex),
            v01=partial(self.forward_corev2, force_fp32=True, SelectiveScan=SelectiveScanMamba),
        )

        self.forward_core = FORWARD_TYPES.get(forward_type, None)
        k_group = 4

        d_proj = d_inner if self.disable_z else (d_inner * 2)
        self.in_proj = nn.Linear(d_model, d_proj, bias=bias, **factory_kwargs)
        self.act: nn.Module = act_layer()

        if d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=d_inner,
                out_channels=d_inner,
                groups=d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False, **factory_kwargs)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.out_proj = nn.Linear(d_inner, d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v0"]:
            self.dt_projs = [
                self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
                for _ in range(k_group)
            ]
            self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
            self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
            del self.dt_projs

            self.A_logs = self.A_log_init(d_state, d_inner, copies=k_group, merge=True)
            self.Ds = self.D_init(d_inner, copies=k_group, merge=True)
        elif initialize in ["v1"]:
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(torch.randn((k_group * d_inner, d_state)))
            self.dt_projs_weight = nn.Parameter(torch.randn((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(torch.randn((k_group, d_inner)))
        elif initialize in ["v2"]:
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(torch.zeros((k_group * d_inner, d_state)))
            self.dt_projs_weight = nn.Parameter(torch.randn((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(torch.randn((k_group, d_inner)))

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor, to_dtype=False, channel_first=False):
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScanCore.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, False)

        if not channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        B, D, H, W = x.shape
        D, N = self.A_logs.shape
        K, D, R = self.dt_projs_weight.shape
        L = H * W

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float()
        Cs = Cs.float()

        As = -torch.exp(self.A_logs.float())
        Ds = self.Ds.float()
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = selective_scan(
            xs, dts,
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = y.transpose(dim0=1, dim1=2).contiguous()
        y = self.out_norm(y).view(B, H, W, -1)

        return (y.to(x.dtype) if to_dtype else y)

    def forward_corev2(self, x: torch.Tensor, channel_first=False, SelectiveScan=SelectiveScanOflex, cross_selective_scan=cross_selective_scan, force_fp32=None):
        if not channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        x = cross_selective_scan(
            x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds, delta_softplus=True,
            out_norm=getattr(self, "out_norm", None),
            out_norm_shape=getattr(self, "out_norm_shape", "v0"),
            force_fp32=force_fp32,
            SelectiveScan=SelectiveScan,
        )
        return x

    def forward(self, x: torch.Tensor, **kwargs):
        with_dconv = (self.d_conv > 1)
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=-1)
            if not self.disable_z_act:
                z = self.act(z)
        if with_dconv:
            x = x.permute(0, 3, 1, 2).contiguous()
            x = self.conv2d(x)
        x = self.act(x)
        y = self.forward_core(x, channel_first=with_dconv)
        if not self.disable_z:
            y = y * z
        out = self.dropout(self.out_proj(y))
        return out



# VSSBlock
class VSSBlock(nn.Module):
    def __init__(self, dim , mlp_ratio=4., drop=0.):
        super(VSSBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = OSSM(d_model=dim, d_state=16, ssm_ratio=2.0, act_layer=nn.SiLU)
        self.drop_path = nn.Dropout(drop)

    def forward(self, x):
        # Check and adjust the input dimensions
        if x.dim() == 3:
            x = x.unsqueeze(0)  # Add batch dimension
        elif x.dim() == 2:
            raise RuntimeError("Input tensor must be at least 3D.")

        x = x + self.drop_path(self.attn(self.norm1(x)))
        # x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionMamba(nn.Module):
    def __init__(self, img_size=64, patch_size=16, in_channels=256, embed_dim=512, depth=12):
        super(VisionMamba, self).__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        # Remove code related to classification token
        self.pos_embed = nn.Parameter(torch.zeros(1, (img_size // patch_size) ** 2, embed_dim))
        self.pos_drop = nn.Dropout(p=0.)

        self.blocks = nn.ModuleList([
            VSSBlock(
                dim=embed_dim
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Batch size
        # b = x.shape[0]
        # Ensure x has 4 dimensions
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = self.patch_embed(x)

        # Remove classification token part
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x

class MambaUNet(nn.Module):
    def __init__(self, num_classes=1, in_channels=4, img_size=256, embed_dim=512, depth=12):
        super(MambaUNet, self).__init__()

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.vim = VisionMamba(img_size=64, patch_size=16, in_channels=256, embed_dim=embed_dim, depth=depth)
        self.encoder1 = ResConvBlock(in_channels, 64)
        self.encoder2 = ResConvBlock(64, 128)
        self.encoder3 = ResConvBlock(128, 256)

        self.upconv3 = UpConv(512, 256)

        self.decoder3 = ResConvBlock(512, 256)

        self.upconv2 = UpConv(256, 128)

        self.decoder2 = ResConvBlock(256, 128)

        self.upconv1 = UpConv(128, 64)

        self.decoder1 = ResConvBlock(128, 64)

        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):

        e1 = self.encoder1(x)  # Output: 256x256x64
        p1 = self.max_pool(e1)  # Output: 128x128x64

        e2 = self.encoder2(p1)  # Output: 128x128x128
        p2 = self.max_pool(e2)  # Output: 64x64x128

        e3 = self.encoder3(p2)  # Output: 64x64x256

        vim_out = self.vim(e3)

        # Reshape the output
        vim_out = vim_out.squeeze(0)
        # Ensure vim_out has the correct dimensions
        b, n, c = vim_out.size()
        # Compute reasonable h and w
        h = w = int(n ** 0.5)
        vim_out = vim_out.view(b, c, h, w) # (batch_size , embedding_dim ,img_size/patch_size ,img_size/patch_size)

        vim_out = F.interpolate(vim_out, size=(32, 32), mode='bilinear', align_corners=False)

        d3 = self.upconv3(vim_out)

        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.decoder3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.decoder2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.decoder1(d1)

        out = self.out_conv(d1)
        return out








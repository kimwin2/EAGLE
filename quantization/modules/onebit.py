import torch
import torch.nn as nn

from quantization.utils.binary_packer import binary_packer, binary_unpacker


class OneBitLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

    def __quant_convert__(self, do_train, quant_func, **kwargs):
        self.quant_func = quant_func
        self.layernorm = nn.LayerNorm(self.out_features)
        self._binarized = False

        if do_train:
            dtype = self.weight.data.dtype
            W = self.weight.data.float()
            original_device = W.device
            calc_device = torch.device("cuda") if torch.cuda.is_available() else original_device

            W_calc = W.to(calc_device)

            U, S, Vh = torch.linalg.svd(torch.abs(W_calc), full_matrices=False)

            sqrt_S_diag = torch.sqrt(torch.diag(S))

            out_channel_scale_gpu = (U @ (sqrt_S_diag[:, 0:1])).view(-1)
            in_channel_scale_gpu = (sqrt_S_diag[0:1, :] @ Vh).view(-1)

            out_channel_scale = out_channel_scale_gpu.to(device=original_device, dtype=dtype)
            in_channel_scale = in_channel_scale_gpu.to(device=original_device, dtype=dtype)

            del W_calc, U, S, Vh, out_channel_scale_gpu, in_channel_scale_gpu, sqrt_S_diag
        else:
            in_channel_scale = torch.empty(self.in_features)
            out_channel_scale = torch.empty(self.out_features)

        self.register_parameter('in_channel_scale', nn.Parameter(in_channel_scale))
        self.register_parameter('out_channel_scale', nn.Parameter(out_channel_scale))

    def forward(self, x):
        *seqlen, hidden_dim = x.shape
        seqlen.append(self.out_features)
        hidden_output_dim = tuple(seqlen)
        x = x.view(-1, hidden_dim)

        in_channel_scale = self.in_channel_scale
        out_channel_scale = self.out_channel_scale

        hidden_states = ((x * in_channel_scale) @ self.quantize(self.weight.to(x.dtype)).t()) * out_channel_scale
        if self.bias is not None:
            hidden_states += self.bias
        hidden_states = self.layernorm(hidden_states)
        hidden_states = hidden_states.reshape(hidden_output_dim)

        return hidden_states

    def quantize(self, x):
        if self._binarized:
            return x
        return self.quant_func(x)

    def extra_repr(self):
        params = {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "bias": self.bias is not None,
        }

        return ", ".join(f"{key}={value}" for key, value in params.items())

    def pack_weights(self):
        """Pack binary weights using binary_packer."""
        packed_weights = {}

        packed_weights['weight_packed'] = binary_packer(self.weight.data.sign().to(torch.int8))
        packed_weights['weight_shape'] = self.weight.shape

        return packed_weights

    def state_dict(self, *args, **kwargs):
        """Override state_dict to save packed weights."""
        state = super().state_dict(*args, **kwargs)
        state.pop('weight', None)

        packed_weights = self.pack_weights()
        packed_weights['weight_shape'] = torch.tensor(packed_weights['weight_shape'])

        state.update(packed_weights)
        return state
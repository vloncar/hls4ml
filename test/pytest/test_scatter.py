from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.fx import wrap as fx_wrap
from torch_scatter import scatter_add, scatter_max, scatter_min, scatter_mul

import hls4ml

test_root_path = Path(__file__).parent


@fx_wrap
def scatter_add_hls4ml(
    src: torch.Tensor, index: torch.Tensor, dim: int = -1, out: Optional[torch.Tensor] = None, dim_size: Optional[int] = None
) -> torch.Tensor:
    return scatter_add(src, index, dim=dim, out=out, dim_size=dim_size)


@fx_wrap
def scatter_max_hls4ml(
    src: torch.Tensor, index: torch.Tensor, dim: int = -1, out: Optional[torch.Tensor] = None, dim_size: Optional[int] = None
) -> torch.Tensor:
    return scatter_max(src, index, dim=dim, out=out, dim_size=dim_size)


@fx_wrap
def scatter_min_hls4ml(
    src: torch.Tensor, index: torch.Tensor, dim: int = -1, out: Optional[torch.Tensor] = None, dim_size: Optional[int] = None
) -> torch.Tensor:
    return scatter_min(src, index, dim=dim, out=out, dim_size=dim_size)


@fx_wrap
def scatter_mul_hls4ml(
    src: torch.Tensor, index: torch.Tensor, dim: int = -1, out: Optional[torch.Tensor] = None, dim_size: Optional[int] = None
) -> torch.Tensor:
    return scatter_mul(src, index, dim=dim, out=out, dim_size=dim_size)


op_map = {
    'sum': scatter_add_hls4ml,
    'max': scatter_max_hls4ml,
    'min': scatter_min_hls4ml,
    'mul': scatter_mul_hls4ml,
}


# Unfortunately, this doesn't work (can't trace with torch.fx). Keeping it just as a reminder of the working coding style.
class ScatterModel(nn.Module):
    def __init__(self, op_str, dim=-1):
        super().__init__()
        self.op = op_map[op_str]
        self.dim = dim

    def forward(self, src, index, out):
        return self.op(src, index, out=out, dim=self.dim)


class ScatterSumModel(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, src, index, out):
        return scatter_add_hls4ml(src, index, out=out, dim=self.dim)


class ScatterMaxModel(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, src, index, out):
        return scatter_max_hls4ml(src, index, out=out, dim=self.dim)


class ScatterMinModel(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, src, index, out):
        return scatter_min_hls4ml(src, index, out=out, dim=self.dim)


class ScatterMulModel(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, src, index, out):
        return scatter_mul_hls4ml(src, index, out=out, dim=self.dim)


model_map = {
    'sum': ScatterSumModel,
    'max': ScatterMaxModel,
    'min': ScatterMinModel,
    'mul': ScatterMulModel,
}


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis'])
@pytest.mark.parametrize('op', ['sum', 'max', 'min', 'mul'])
@pytest.mark.parametrize('dim', [0, 1])
@pytest.mark.parametrize('io_type', ['io_parallel'])
def test_scatter_2d(backend, op, dim, io_type):
    '''
    Test proper handling of PyG's scatter_* operation.
    '''
    # model = ScatterModel(op, dim)
    model = model_map[op](dim=dim)
    model.eval()

    if dim == 0:
        src = np.asarray([[2, 0], [1, 4], [3, 0], [2, 1], [3, 4]], dtype=np.float32)
        index = np.asarray([[4, 5], [4, 2], [3, 0], [0, 2], [2, 1]], dtype=np.float32)
        out = np.ones((6, 2), dtype=src.dtype)
    else:
        src = np.asarray([[2, 0, 1, 4, 3], [0, 2, 1, 3, 4]], dtype=np.float32)
        index = np.asarray([[4, 5, 4, 2, 3], [0, 0, 2, 2, 1]], dtype=np.float32)
        out = np.ones((2, 6), dtype=src.dtype)

    src_t = torch.tensor(src)
    index_t = torch.tensor(index, dtype=torch.int64)
    out_t = torch.tensor(out)

    pytorch_result = model(src_t, index_t, out_t)
    if op in ['min', 'max']:
        # scatter_min and scatter_max return the tuple (result, argmin/max)
        pytorch_result, pytorch_argminmax = pytorch_result
        pytorch_result = pytorch_result.detach().numpy().flatten()
        pytorch_argminmax = pytorch_argminmax.detach().numpy().flatten()
    else:
        pytorch_result = pytorch_result.detach().numpy().flatten()

    config = hls4ml.utils.config_from_pytorch_model(model, inputs_channel_last=None, transpose_outputs=False)
    config['Model'].pop('InputsChannelLast')
    output_dir = str(test_root_path / f'hls4mlprj_pyg_scatter_{op}_{dim}_{backend}_{io_type}')

    hls_model = hls4ml.converters.convert_from_pytorch_model(
        model,
        [(None,) + src.shape, (None,) + index.shape, (None,) + out.shape],
        hls_config=config,
        output_dir=output_dir,
        backend=backend,
        io_type=io_type,
    )
    hls_model.compile()

    hls_result = hls_model.predict([src, index, out])

    np.testing.assert_allclose(hls_result, pytorch_result, rtol=1e-2, atol=0.01)

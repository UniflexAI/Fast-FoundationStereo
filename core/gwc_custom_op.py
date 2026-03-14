"""
Custom ONNX op for build_gwc_volume. Exports as BuildGwcVolume node for TensorRT plugin.

Uses ONNX-safe implementation (no unfold) so the tracer can run it; the symbolic
replaces it with a single BuildGwcVolume node for TensorRT.
"""
import torch
import torch.nn.functional as F


def _build_gwc_volume_onnx_safe(
    refimg_fea: torch.Tensor,
    targetimg_fea: torch.Tensor,
    maxdisp: int,
    num_groups: int,
    normalize: bool = True,
) -> torch.Tensor:
    """ONNX-compatible implementation (no unfold). Same logic as build_gwc_volume_optimized_pytorch1."""
    dtype = refimg_fea.dtype
    B, C, H, W = refimg_fea.shape
    channels_per_group = C // num_groups

    ref_volume = refimg_fea.unsqueeze(2).expand(B, C, maxdisp, H, W)
    padded_target = F.pad(targetimg_fea, (maxdisp - 1, 0, 0, 0))

    # Replace unfold with loop of slices (ONNX supports Slice)
    slices = []
    for d in range(maxdisp):
        start = maxdisp - 1 - d
        t = padded_target[:, :, :, start : start + W]
        slices.append(t)
    target_volume = torch.stack(slices, dim=2)

    ref_volume = ref_volume.view(B, num_groups, channels_per_group, maxdisp, H, W)
    target_volume = target_volume.view(B, num_groups, channels_per_group, maxdisp, H, W)
    if normalize:
        ref_volume = F.normalize(ref_volume.float(), dim=2).to(dtype)
        target_volume = F.normalize(target_volume.float(), dim=2).to(dtype)

    cost_volume = (ref_volume * target_volume).sum(dim=2)
    return cost_volume.contiguous()


@torch.library.custom_op("ffs::build_gwc_volume", mutates_args=())
def build_gwc_volume_custom(
    ref: torch.Tensor,
    tgt: torch.Tensor,
    maxdisp: int,
    num_groups: int,
    normalize: bool = True,
) -> torch.Tensor:
    """Custom op that exports as BuildGwcVolume ONNX node. Use for single-ONNX export."""
    return _build_gwc_volume_onnx_safe(ref, tgt, maxdisp, num_groups, normalize)


@build_gwc_volume_custom.register_fake
def _(ref, tgt, maxdisp, num_groups, normalize=True):
    B, C, H, W = ref.shape
    return torch.empty(B, num_groups, maxdisp, H, W, dtype=ref.dtype, device=ref.device)


def _gwc_onnx_symbolic(g, ref, tgt, maxdisp, num_groups, normalize):
    """ONNX symbolic: create BuildGwcVolume node for TensorRT plugin."""
    from torch.onnx.symbolic_helper import _parse_arg
    maxdisp_val = _parse_arg(maxdisp, "i")
    num_groups_val = _parse_arg(num_groups, "i")
    return g.op(
        "BuildGwcVolume",
        ref,
        tgt,
        maxdisp_i=maxdisp_val,
        num_groups_i=num_groups_val,
        plugin_namespace_s="",
    )


def register_gwc_onnx_symbolic():
    """Register ONNX symbolic for ffs::build_gwc_volume. Call before torch.onnx.export."""
    torch.onnx.register_custom_op_symbolic(
        "ffs::build_gwc_volume",
        _gwc_onnx_symbolic,
        opset_version=17,
    )

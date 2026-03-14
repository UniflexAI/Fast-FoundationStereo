"""
TensorRT 10 IPluginV3 plugin for group-wise correlation (GWC) volume.

The plugin computes:
    gwc[b, g, d, h, w] = sum_{c in group g} ref[b, c, h, w] * tgt[b, c, h, w-d]

Inputs:  ref (B, C, H, W) fp32 or fp16
         tgt (B, C, H, W) fp32 or fp16
Output:  gwc (B, G, D, H, W) fp16

Attributes (serialized):
    maxdisp    (int32) – number of disparity levels D
    num_groups (int32) – number of channel groups G

Usage:
    from plugins.gwc_plugin import register_gwc_plugin
    register_gwc_plugin()   # before building or deserializing any engine
"""

import ctypes
import platform

import numpy as np
import tensorrt as trt

# ---------------------------------------------------------------------------
# CUDA kernel source (fp32 and fp16 input variants, always fp16 output)
# ---------------------------------------------------------------------------
_KERNEL_SRC = r"""
#include <cuda_fp16.h>

extern "C" __global__ void build_gwc_f32(
    const float* __restrict__ ref, const float* __restrict__ tgt,
    __half* __restrict__ gwc,
    int B, int C, int H, int W, int D, int G
) {
    int w   = blockIdx.x * blockDim.x + threadIdx.x;
    int h   = blockIdx.y * blockDim.y + threadIdx.y;
    int bgd = blockIdx.z;
    if (w >= W || h >= H || bgd >= B * G * D) return;
    int d = bgd % D, g = (bgd / D) % G, b = bgd / (D * G);
    int Cg = C / G, c0 = g * Cg, tw = w - d;
    float sum = 0.0f;
    for (int c = 0; c < Cg; ++c) {
        int base = b*C*H*W + (c0+c)*H*W + h*W;
        sum += ref[base + w] * ((tw >= 0) ? tgt[base + tw] : 0.0f);
    }
    gwc[b*G*D*H*W + g*D*H*W + d*H*W + h*W + w] = __float2half(sum);
}

extern "C" __global__ void build_gwc_f16(
    const __half* __restrict__ ref, const __half* __restrict__ tgt,
    __half* __restrict__ gwc,
    int B, int C, int H, int W, int D, int G
) {
    int w   = blockIdx.x * blockDim.x + threadIdx.x;
    int h   = blockIdx.y * blockDim.y + threadIdx.y;
    int bgd = blockIdx.z;
    if (w >= W || h >= H || bgd >= B * G * D) return;
    int d = bgd % D, g = (bgd / D) % G, b = bgd / (D * G);
    int Cg = C / G, c0 = g * Cg, tw = w - d;
    float sum = 0.0f;
    for (int c = 0; c < Cg; ++c) {
        int base = b*C*H*W + (c0+c)*H*W + h*W;
        sum += __half2float(ref[base + w])
             * ((tw >= 0) ? __half2float(tgt[base + tw]) : 0.0f);
    }
    gwc[b*G*D*H*W + g*D*H*W + d*H*W + h*W + w] = __float2half(sum);
}
"""

_kernel_cache: dict = {}  # dtype_str -> (fn_f32, fn_f16)


def _get_kernels():
    if _kernel_cache:
        return _kernel_cache["fns"]
    from cuda import nvrtc, cuda as cuda_drv

    cuda_drv.cuInit(0)
    _, dev = cuda_drv.cuDeviceGet(0)
    _, major = cuda_drv.cuDeviceGetAttribute(
        cuda_drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev)
    _, minor = cuda_drv.cuDeviceGetAttribute(
        cuda_drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev)

    arch = f"--gpu-architecture=compute_{major}{minor}".encode()
    cpu_arch = "aarch64-linux" if "aarch64" in platform.machine() else "x86_64-linux"
    include = f"--include-path=/usr/local/cuda/targets/{cpu_arch}/include".encode()

    _, prog = nvrtc.nvrtcCreateProgram(_KERNEL_SRC.encode(), b"gwc_plugin.cu", 0, [], [])
    opts = [arch, include]
    err = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    if err[0] != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        _, logsize = nvrtc.nvrtcGetProgramLogSize(prog)
        log = b" " * logsize
        nvrtc.nvrtcGetProgramLog(prog, log)
        raise RuntimeError(f"GWC plugin NVRTC compile failed:\n{log.decode()}")

    _, ptx_size = nvrtc.nvrtcGetPTXSize(prog)
    ptx = b" " * ptx_size
    nvrtc.nvrtcGetPTX(prog, ptx)

    _, ctx = cuda_drv.cuDevicePrimaryCtxRetain(dev)
    cuda_drv.cuCtxPushCurrent(ctx)
    _, module = cuda_drv.cuModuleLoadData(ptx)
    _, fn_f32 = cuda_drv.cuModuleGetFunction(module, b"build_gwc_f32")
    _, fn_f16 = cuda_drv.cuModuleGetFunction(module, b"build_gwc_f16")
    _kernel_cache["fns"] = (fn_f32, fn_f16)
    return fn_f32, fn_f16


def _launch_gwc(fn, stream, ref_ptr, tgt_ptr, gwc_ptr, B, C, H, W, D, G):
    from cuda import cuda as cuda_drv
    BX, BY = 16, 16
    c_ref = ctypes.c_uint64(ref_ptr)
    c_tgt = ctypes.c_uint64(tgt_ptr)
    c_gwc = ctypes.c_uint64(gwc_ptr)
    c_B, c_C, c_H, c_W, c_D, c_G = (ctypes.c_int(x) for x in (B, C, H, W, D, G))
    args = (ctypes.c_void_p * 9)(
        ctypes.addressof(c_ref), ctypes.addressof(c_tgt), ctypes.addressof(c_gwc),
        ctypes.addressof(c_B),   ctypes.addressof(c_C),   ctypes.addressof(c_H),
        ctypes.addressof(c_W),   ctypes.addressof(c_D),   ctypes.addressof(c_G),
    )
    cuda_drv.cuLaunchKernel(
        fn,
        (W + BX - 1) // BX, (H + BY - 1) // BY, B * G * D,
        BX, BY, 1,
        0, stream, args, 0,
    )


# ---------------------------------------------------------------------------
# TRT Plugin
# ---------------------------------------------------------------------------

class GwcVolumePlugin(
    trt.IPluginV3,
    trt.IPluginV3OneCore,
    trt.IPluginV3OneBuild,
    trt.IPluginV3OneRuntime,
):
    PLUGIN_NAME    = "BuildGwcVolume"
    PLUGIN_VERSION = "1"

    def __init__(self, maxdisp: int = 48, num_groups: int = 8, phase=None):
        trt.IPluginV3.__init__(self)
        trt.IPluginV3OneCore.__init__(self)
        trt.IPluginV3OneBuild.__init__(self)
        trt.IPluginV3OneRuntime.__init__(self)

        self.maxdisp       = int(maxdisp)
        self.num_groups    = int(num_groups)
        self.plugin_namespace = ""
        self.plugin_name      = self.PLUGIN_NAME
        self.plugin_version   = self.PLUGIN_VERSION
        self.num_outputs      = 1
        self.timing_cache_id  = ""

        # Runtime shape, filled by on_shape_change
        self.B = 1
        self.C = 0
        self.H = 0
        self.W = 0
        self._use_fp16_input = False

        if phase is not None:
            self.phase = phase

    # ---- IPluginV3 --------------------------------------------------------
    def get_capability_interface(self, cap_type):
        return self

    def clone(self):
        c = GwcVolumePlugin(self.maxdisp, self.num_groups)
        c.__dict__.update(self.__dict__)
        return c

    # ---- IPluginV3OneBuild ------------------------------------------------
    def get_output_data_types(self, input_types):
        return [trt.DataType.HALF]

    def get_output_shapes(self, inputs, shape_inputs, expr_builder):
        # inputs[0]: ref (B, C, H, W) → output (B, G, D, H, W)
        out = trt.DimsExprs(inputs[0])   # copy 4-dim → we'll extend to 5
        # Build a 5-element DimsExprs by composing expressions
        B = inputs[0][0]
        H = inputs[0][2]
        W = inputs[0][3]
        G = expr_builder.constant(self.num_groups)
        D = expr_builder.constant(self.maxdisp)
        result = trt.DimsExprs(5)
        result[0] = B
        result[1] = G
        result[2] = D
        result[3] = H
        result[4] = W
        return [result]

    def get_fields_to_serialize(self):
        return trt.PluginFieldCollection([
            trt.PluginField("maxdisp",
                            np.array([self.maxdisp],    dtype=np.int32),
                            trt.PluginFieldType.INT32),
            trt.PluginField("num_groups",
                            np.array([self.num_groups], dtype=np.int32),
                            trt.PluginFieldType.INT32),
        ])

    def configure_plugin(self, inp, out):
        self._use_fp16_input = (inp[0].desc.type == trt.DataType.HALF)

    def supports_format_combination(self, pos, in_out, num_inputs):
        desc = in_out[pos].desc
        if desc.format != trt.TensorFormat.LINEAR:
            return False
        if pos < 2:   # inputs: fp32 or fp16
            return desc.type in (trt.DataType.FLOAT, trt.DataType.HALF)
        # output: always fp16
        return desc.type == trt.DataType.HALF

    def get_valid_tactics(self):
        return [1]   # 0 is reserved by TRT as the "default" tactic

    def set_tactic(self, tactic):
        pass

    # ---- IPluginV3OneRuntime -----------------------------------------------
    def on_shape_change(self, inp, out):
        dims = inp[0].dims
        self.B = dims[0]
        self.C = dims[1]
        self.H = dims[2]
        self.W = dims[3]
        self._use_fp16_input = (inp[0].type == trt.DataType.HALF)

    def enqueue(self, input_desc, output_desc, inputs, outputs, workspace, stream):
        fn_f32, fn_f16 = _get_kernels()
        fn = fn_f16 if input_desc[0].type == trt.DataType.HALF else fn_f32
        _launch_gwc(
            fn, stream,
            inputs[0], inputs[1], outputs[0],
            self.B, self.C, self.H, self.W, self.maxdisp, self.num_groups,
        )
        return 0

    def attach_to_context(self, context):
        return self.clone()


# ---------------------------------------------------------------------------
# Plugin Creator
# ---------------------------------------------------------------------------

class GwcVolumePluginCreator(trt.IPluginCreatorV3One):
    def __init__(self):
        trt.IPluginCreatorV3One.__init__(self)
        self.name             = GwcVolumePlugin.PLUGIN_NAME
        self.plugin_namespace = ""
        self.plugin_version   = GwcVolumePlugin.PLUGIN_VERSION
        self.field_names = trt.PluginFieldCollection([
            trt.PluginField("maxdisp",    np.array([], dtype=np.int32),
                            trt.PluginFieldType.INT32),
            trt.PluginField("num_groups", np.array([], dtype=np.int32),
                            trt.PluginFieldType.INT32),
        ])

    def create_plugin(self, name, fc, phase):
        maxdisp    = 48
        num_groups = 8
        for f in fc:
            if f.name == "maxdisp":
                maxdisp = int(f.data[0])
            elif f.name == "num_groups":
                num_groups = int(f.data[0])
        return GwcVolumePlugin(maxdisp=maxdisp, num_groups=num_groups, phase=phase)


# ---------------------------------------------------------------------------
# Registration helper
# ---------------------------------------------------------------------------

_registered = False


def register_gwc_plugin() -> None:
    """Register the GWC plugin creator with TRT's global plugin registry.
    Must be called before building or deserializing any engine that uses this plugin.
    Safe to call multiple times.
    """
    global _registered
    if not _registered:
        trt.get_plugin_registry().register_creator(GwcVolumePluginCreator(), "")
        _registered = True

#!/usr/bin/env python3
"""
Build a single TensorRT engine for FoundationStereo from foundation_stereo.onnx.

The ONNX must contain a BuildGwcVolume custom op node. Use the C++ GWC plugin
(build with scripts/build_gwc_plugin.sh) so the ONNX parser can find it.

Usage:
    # Build the C++ plugin first:
    ./scripts/build_gwc_plugin.sh

    # Build engine (requires --plugin for C++ plugin):
    python scripts/build_engine.py \
        --onnx  output/480x864/foundation_stereo.onnx \
        --engine output/480x864/foundation_stereo.engine \
        --plugin plugins/gwc_plugin_cpp/build/libGwcVolumePlugin.so

    # FP32 mode (slower, for debugging):
    python scripts/build_engine.py --onnx ... --engine ... --plugin ... --fp32
"""
import argparse
import ctypes
import os
import sys
from typing import Optional

code_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(code_dir, ".."))

import tensorrt as trt


def _load_plugin_so(plugin_path: str) -> None:
    """Load C++ plugin .so so REGISTER_TENSORRT_PLUGIN runs and registers the plugin."""
    path = os.path.abspath(plugin_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Plugin not found: {path}\nRun: ./scripts/build_gwc_plugin.sh")
    ctypes.CDLL(path)


def _patch_onnx(onnx_path: str) -> bytes:
    """Patch ONNX: plugin_namespace for BuildGwcVolume; Squeeze in If else_branch so both branches output 4D."""
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper
    model = onnx.load(onnx_path)
    for node in model.graph.node:
        if node.op_type == "BuildGwcVolume":
            has_ns = any(a.name == "plugin_namespace" for a in node.attribute)
            if not has_ns:
                node.attribute.append(helper.make_attribute("plugin_namespace", ""))
        if node.op_type == "If" and "/post_runner/If" in node.name:
            for attr in node.attribute:
                if attr.name == "else_branch":
                    g = attr.g
                    out_name = g.output[0].name
                    new_out = out_name + "_4d"
                    axes_name = "axes_squeeze_1"
                    axes_init = numpy_helper.from_array(np.array([1], dtype=np.int64), axes_name)
                    g.initializer.append(axes_init)
                    squeeze = helper.make_node("Squeeze", [out_name, axes_name], [new_out])
                    g.node.append(squeeze)
                    g.output[0].name = new_out
                    break
            break
    return model.SerializeToString()


def build_engine(onnx_path: str, engine_path: str, fp16: bool = True, plugin_path: Optional[str] = None) -> None:
    if plugin_path:
        _load_plugin_so(plugin_path)

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)

    onnx_bytes = _patch_onnx(onnx_path)
    if not parser.parse(onnx_bytes):
        for i in range(parser.num_errors):
            print("ONNX parse error:", parser.get_error(i))
        raise RuntimeError(f"Failed to parse ONNX: {onnx_path}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 512 << 20)  # 512 MB
    config.builder_optimization_level = 3  # 0=disabled, 3=moderate, 4-5=heavier (slower build)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    print(f"Building engine from {onnx_path}  (fp16={fp16}) …")
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError("Engine build failed — check TRT warnings above.")

    os.makedirs(os.path.dirname(os.path.abspath(engine_path)) or ".", exist_ok=True)
    engine_data = bytes(engine_bytes)
    with open(engine_path, "wb") as f:
        f.write(engine_data)
    print(f"Engine saved → {engine_path}  ({len(engine_data) // 1024 // 1024} MB)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--onnx",   required=True, help="path to foundation_stereo.onnx")
    p.add_argument("--engine", required=True, help="output .engine path")
    p.add_argument("--plugin", required=True, help="path to libGwcVolumePlugin.so (from build_gwc_plugin.sh)")
    p.add_argument("--fp32", action="store_true", help="disable fp16 (slower)")
    args = p.parse_args()
    build_engine(args.onnx, args.engine, fp16=not args.fp32, plugin_path=args.plugin)

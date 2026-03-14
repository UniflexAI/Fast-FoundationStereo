#!/usr/bin/env python3
"""
Run inference and profile speed for foundation_stereo.engine (single TRT engine).

Usage:
    python scripts/run_inference.py \
        --engine output/480x864/foundation_stereo.engine \
        --plugin plugins/gwc_plugin_cpp/build/libGwcVolumePlugin.so \
        [--left demo_data/left.png --right demo_data/right.png] \
        [--out_dir output/480x864/result] \
        [--warmup 10 --iters 50]
"""
import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import argparse
import ctypes
import os
import sys
import time

code_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(code_dir, ".."))

import numpy as np
import torch


def _load_plugin(plugin_path: str) -> None:
    path = os.path.abspath(plugin_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Plugin not found: {path}")
    ctypes.CDLL(path)


def load_engine(engine_path: str, plugin_path: str):
    import tensorrt as trt
    _load_plugin(plugin_path)
    with open(engine_path, "rb") as f:
        engine_data = f.read()
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_data)
    context = engine.create_execution_context()
    return engine, context


def run_inference(engine, context, left: torch.Tensor, right: torch.Tensor):
    import tensorrt as trt
    # Engine expects left, right; outputs disp
    in_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)
                if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT]
    out_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)
                 if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT]

    # FP16 engine expects half
    for name in in_names:
        dt = engine.get_tensor_dtype(name)
        if dt == trt.DataType.HALF and left.dtype != torch.float16:
            left = left.half()
            right = right.half()
        break

    context.set_input_shape("left", tuple(left.shape))
    context.set_input_shape("right", tuple(right.shape))
    context.set_tensor_address("left", int(left.data_ptr()))
    context.set_tensor_address("right", int(right.data_ptr()))

    out_shape = tuple(context.get_tensor_shape("disp"))
    out_dtype = torch.float16 if engine.get_tensor_dtype("disp") == trt.DataType.HALF else torch.float32
    disp = torch.empty(out_shape, device="cuda", dtype=out_dtype)
    context.set_tensor_address("disp", int(disp.data_ptr()))

    stream = torch.cuda.current_stream().cuda_stream
    ok = context.execute_async_v3(stream)
    assert ok, "TRT execute failed"
    return disp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True, help="foundation_stereo.engine path")
    parser.add_argument("--plugin", required=True, help="libGwcVolumePlugin.so path")
    parser.add_argument("--left", default=None, help="left image (optional)")
    parser.add_argument("--right", default=None, help="right image (optional)")
    parser.add_argument("--out_dir", default=None, help="save disp and vis here")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=864)
    parser.add_argument("--compare", action="store_true", help="compare with PyTorch (needs model weights)")
    args = parser.parse_args()

    engine, context = load_engine(args.engine, args.plugin)

    # Create input (H, W must match engine)
    H, W = args.height, args.width
    if args.left and args.right:
        import imageio
        img0 = np.asarray(imageio.imread(args.left))
        img1 = np.asarray(imageio.imread(args.right))
        if len(img0.shape) == 2:
            img0 = np.tile(img0[..., None], (1, 1, 3))
            img1 = np.tile(img1[..., None], (1, 1, 3))
        img0 = img0[..., :3]
        img1 = img1[..., :3]
        import cv2
        img0 = cv2.resize(img0, (W, H))
        img1 = cv2.resize(img1, (W, H))
        left = torch.from_numpy(img0).cuda().float().permute(2, 0, 1)[None]
        right = torch.from_numpy(img1).cuda().float().permute(2, 0, 1)[None]
    else:
        left = torch.randn(1, 3, H, W, device="cuda", dtype=torch.float32) * 128 + 128
        right = torch.randn(1, 3, H, W, device="cuda", dtype=torch.float32) * 128 + 128

    left = left.clamp(0, 255)
    right = right.clamp(0, 255)

    # Warmup
    for _ in range(args.warmup):
        disp = run_inference(engine, context, left, right)
    torch.cuda.synchronize()

    # Profile
    times = []
    for i in range(args.iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        disp = run_inference(engine, context, left, right)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    avg_ms = np.mean(times)
    std_ms = np.std(times)
    print(f"TensorRT inference: {avg_ms:.1f} ± {std_ms:.1f} ms  ({args.iters} iters, {H}x{W})")
    print(f"  FPS: {1000/avg_ms:.1f}")

    if args.compare:
        # Compare with PyTorch TrtFullRunner
        model_dir = os.path.join(code_dir, "../weights/20-30-48/model_best_bp2_serialize.pth")
        if os.path.isfile(model_dir):
            from core.foundation_stereo import TrtFullRunner
            from core.gwc_custom_op import build_gwc_volume_custom
            model = torch.load(model_dir, map_location="cpu", weights_only=False)
            model.args.valid_iters = 4
            model.args.max_disp = 192
            model.cuda().eval()
            runner = TrtFullRunner(model).cuda().eval()
            with torch.no_grad():
                disp_pt = runner(left, right)
            disp_pt = disp_pt[0, 0].cpu().float().numpy().clip(0, None)
            disp_trt = disp[0, 0].cpu().float().numpy().clip(0, None)
            mae = np.abs(disp_pt - disp_trt).mean()
            max_diff = np.abs(disp_pt - disp_trt).max()
            print(f"vs PyTorch: MAE={mae:.4f}, max_diff={max_diff:.4f}")
        else:
            print(f"Skip compare: model not found at {model_dir}")

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        disp_np = disp[0, 0].cpu().float().numpy().clip(0, None)
        np.save(os.path.join(args.out_dir, "disp.npy"), disp_np)
        try:
            import cv2
            from Utils import vis_disparity
            vis = vis_disparity(disp_np, color_map=cv2.COLORMAP_TURBO)
            import imageio
            imageio.imwrite(os.path.join(args.out_dir, "disp_vis.png"), vis)
            print(f"Saved to {args.out_dir}/")
        except Exception as e:
            print(f"Could not save vis: {e}")


if __name__ == "__main__":
    main()

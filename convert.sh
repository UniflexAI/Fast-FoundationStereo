python scripts/run_demo.py \
   --model_dir weights/20-30-48/model_best_bp2_serialize.pth \
   --left_file test_data/left.png \
   --right_file test_data/right.png \
   --intrinsic_file test_data/K.txt \
   --out_dir output/pytorch_inference_480x864/ \
   --remove_invisible 0 \
   --denoise_cloud 0 \
   --scale 1 \
   --get_pc 0 \
   --valid_iters 4 \
   --max_disp 64 \
   --zfar 100

# 480x848 input, padded to 480x864 (nearest multiple of 32)
# make_onnx.py exports feature_runner.onnx and post_runner.onnx (separate, for two-engine TRT)
python scripts/make_onnx.py \
  --model_dir weights/20-30-48/model_best_bp2_serialize.pth \
  --save_path output/480x864 \
  --height 480 \
  --width 864 \
  --valid_iters 4 \
  --max_disp 64

#trtexec --onnx=output/480x864/feature_runner.onnx --saveEngine=output/480x864/feature_runner.engine  --useCudaGraph
#trtexec --onnx=output/480x864/post_runner.onnx --saveEngine=output/480x864/post_runner.engine   --useCudaGraph

# python scripts/run_demo_tensorrt.py \
#   --onnx_dir output/480x864/ \
#   --left_file test_data/left.png \
#   --right_file test_data/right.png \
#   --intrinsic_file test_data/K.txt \
#   --out_dir output/trt_inference_480x864/ \
#   --remove_invisible 0 \
#   --denoise_cloud 0 \
#   --get_pc 0 \
#   --zfar 100 

# single ONNX export (BuildGwcVolume custom plugin op)
python scripts/make_onnx.py \
  --model_dir weights/20-30-48/model_best_bp2_serialize.pth \
  --save_path output/480x864 \
  --height 480 \
  --width 864 \
  --valid_iters 4 \
  --max_disp 64 \
  --single

export PATH=/usr/local/cuda-12.8/bin:$PATH
nvcc --version
# build GWC TensorRT plugin (libGwcVolumePlugin.so)
(cd plugins/gwc_plugin_cpp && mkdir -p build && cd build && cmake .. && make -j)

# single-engine TensorRT build (requires plugin .so at plugins/gwc_plugin_cpp/build/libGwcVolumePlugin.so)
trtexec \
  --onnx=output/480x864/foundation_stereo.onnx \
  --saveEngine=output/480x864/foundation_stereo.engine \
  --fp16 \
  --plugins=plugins/gwc_plugin_cpp/build/libGwcVolumePlugin.so \
  --useCudaGraph

# run single-engine TensorRT inference
python scripts/run_demo_tensorrt_single.py \
  --engine output/480x864/foundation_stereo.engine \
  --plugin_lib plugins/gwc_plugin_cpp/build/libGwcVolumePlugin.so \
  --onnx_dir output/480x864 \
  --left_file test_data/left.png \
  --right_file test_data/right.png \
  --intrinsic_file test_data/K.txt \
  --out_dir output/trt_single_inference_480x864/ \
  --remove_invisible 0 \
  --denoise_cloud 0 \
  --get_pc 0 \
  --zfar 100
srcs = [
-        "openvino/runtime/lib/intel64/Release/openvino.lib",
-        "openvino/runtime/lib/intel64/Release/openvino_tensorflow_lite_frontend.lib",
+        "openvino/runtime/lib/intel64/Debug/openvinod.lib",
+        "openvino/runtime/lib/intel64/Debug/openvino_tensorflow_lite_frontendd.lib",
     ],

C:\Workspace\junwei\LiteRT>"C:\Program Files (x86)\Intel\openvino_2025.3\setupvars.bat"

set OPENVINO_NATIVE_DIR=C:\Program Files (x86)\Intel\openvino_2025.3
C:\Workspace\junwei\LiteRT>bazelisk-windows-amd64 build -c dbg //litert/tools:benchmark_model

C:\Workspace\junwei\LiteRT>bazelisk-windows-amd64 build -c dbg //litert/vendors/intel_openvino/dispatch:LiteRtDispatch  --config=windows 

C:\Workspace\junwei\LiteRT>bazelisk-windows-amd64 build -c dbg //litert/vendors/intel_openvino/compiler:LiteRtCompilerPlugin  --config=windows 

C:\Workspace\junwei\LiteRT>.\bazel-bin\litert\tools\benchmark_model.exe ^
    --graph=C:\Workspace\junwei\dump_tflite_models\reshape_simple.tflite ^
    --require_full_delegation=false --use_npu=true ^
    --compiler_plugin_library_path=C:\Workspace\junwei\LiteRT\bazel-bin\litert\vendors\intel_openvino\compiler ^
    --dispatch_library_path=C:\Workspace\junwei\LiteRT\bazel-bin\litert\vendors\intel_openvino\dispatch 

> bazelisk-windows-amd64 build -c dbg //litert/tools:run_model

> .\bazel-bin\litert\tools\run_model.exe --accelerator=npu ^
    --compiler_plugin_library_dir=C:\Workspace\junwei\LiteRT\bazel-bin\litert\vendors\intel_openvino\compiler ^
    --dispatch_library_dir=C:\Workspace\junwei\LiteRT\bazel-bin\litert\vendors\intel_openvino\dispatch  ^
    --graph=C:\Workspace\junwei\dump_tflite_models\mobilenet_v2_1.0_224.tflite ^
    --print_tensors=true ^
    --compare_numerical=true

Debug OpenVINO with VS
third_party/intel_openvino/openvino.bazel
     srcs = select({
         "@org_tensorflow//tensorflow:windows": [
-            "openvino/runtime/lib/intel64/Release/openvino.lib",
-            "openvino/runtime/lib/intel64/Release/openvino_tensorflow_lite_frontend.lib",
+            "openvino/runtime/lib/intel64/Debug/openvinod.lib",
+            "openvino/runtime/lib/intel64/Debug/openvino_tensorflow_lite_frontendd.lib",
         ],

Path
C:\Program Files (x86)\Intel\openvino_2025.3\\runtime\3rdparty\tbb\bin;C:\Program Files (x86)\Intel\openvino_2025.3\\runtime\3rdparty\tbb\bin\intel64\vc14;C:\Program Files (x86)\Intel\openvino_2025.3\\runtime\3rdparty\tbb\redist\intel64\vc14;C:\Program Files (x86)\Intel\openvino_2025.3\\runtime\bin\intel64\Release;C:\Program Files (x86)\Intel\openvino_2025.3\\runtime\bin\intel64\Debug

--accelerator=npu --compiler_plugin_library_dir=C:\Workspace\junwei\LiteRT\bazel-bin\litert\vendors\intel_openvino\compiler  --dispatch_library_dir=C:\Workspace\junwei\LiteRT\bazel-bin\litert\vendors\intel_openvino\dispatch   --graph=C:\Workspace\junwei\testing\testdata\one_mul.tflite --print_tensors=true --compare_numerical=true

============================Release========================
C:\Workspace\junwei\LiteRT>"C:\Program Files (x86)\Intel\openvino_2025.3\setupvars.bat"

set OPENVINO_NATIVE_DIR=C:\Program Files (x86)\Intel\openvino_2025.3
C:\Workspace\junwei\LiteRT>bazelisk-windows-amd64 build -c opt //litert/tools:benchmark_model

C:\Workspace\junwei\LiteRT>bazelisk-windows-amd64 build -c opt //litert/vendors/intel_openvino/dispatch:LiteRtDispatch --config=windows

C:\Workspace\junwei\LiteRT>bazelisk-windows-amd64 build -c opt //litert/vendors/intel_openvino/compiler:LiteRtCompilerPlugin --config=windows

# use_npu
C:\Workspace\junwei\LiteRT>.\bazel-bin\litert\tools\benchmark_model.exe ^
    --graph=C:\Workspace\junwei\dump_tflite_models\mobilenet_v2_1.0_224.tflite ^
    --require_full_delegation=false --use_npu=true ^
    --compiler_plugin_library_path=C:\Workspace\junwei\LiteRT\bazel-bin\litert\vendors\intel_openvino\compiler ^
    --dispatch_library_path=C:\Workspace\junwei\LiteRT\bazel-bin\litert\vendors\intel_openvino\dispatch 

========== BENCHMARK RESULTS ==========
INFO: [.\litert/tools/benchmark_litert_model.h:57] Model initialization: 586.76 ms
INFO: [.\litert/tools/benchmark_litert_model.h:59] Warmup (first):       4.44 ms
INFO: [.\litert/tools/benchmark_litert_model.h:61] Warmup (avg):         0.52 ms (957 runs)
INFO: [.\litert/tools/benchmark_litert_model.h:63] Inference (avg):      0.46 ms (2166 runs)
INFO: [.\litert/tools/benchmark_litert_model.h:67] Inference (min):      0.35 ms
INFO: [.\litert/tools/benchmark_litert_model.h:69] Inference (max):      0.74 ms
INFO: [.\litert/tools/benchmark_litert_model.h:71] Inference (std):      0.03
INFO: [.\litert/tools/benchmark_litert_model.h:78] Throughput:           1243.93 MB/s
INFO: [.\litert/tools/benchmark_litert_model.h:101] ======================================

# use_cpu
C:\Workspace\junwei\LiteRT>.\LiteRT\bazel-bin\litert\tools\benchmark_model.exe ^
    --graph=C:\Workspace\junwei\dump_tflite_models\mobilenet_v2_1.0_224.tflite ^
    --require_full_delegation=false --use_cpu=true
========== BENCHMARK RESULTS ==========
INFO: [.\litert/tools/benchmark_litert_model.h:57] Model initialization: 11.03 ms
INFO: [.\litert/tools/benchmark_litert_model.h:59] Warmup (first):       11.50 ms
INFO: [.\litert/tools/benchmark_litert_model.h:61] Warmup (avg):         9.85 ms (51 runs)
INFO: [.\litert/tools/benchmark_litert_model.h:63] Inference (avg):      10.08 ms (100 runs)
INFO: [.\litert/tools/benchmark_litert_model.h:67] Inference (min):      9.20 ms
INFO: [.\litert/tools/benchmark_litert_model.h:69] Inference (max):      16.27 ms
INFO: [.\litert/tools/benchmark_litert_model.h:71] Inference (std):      1.02
INFO: [.\litert/tools/benchmark_litert_model.h:78] Throughput:           56.96 MB/s
INFO: [.\litert/tools/benchmark_litert_model.h:101] ======================================


C:\Workspace\junwei\LiteRT>set PATH="%PATH%;C:\Workspace\junwei\openvino\bin\intel64\Release;C:\Workspace\junwei\openvino\temp\Windows_AMD64\tbb\bin

=============Enable Gemma3 on OpenVINO NPU=====================
> bazelisk-windows-amd64 build -c opt //litert/tools:run_model
> bazelisk-windows-amd64 build //litert/tools:run_model --copt=-DABSL_FLAGS_STRIP_NAMES=0 --copt=-DABSL_FLAGS_STRIP_HELP=0
> .\bazel-bin\litert\tools\run_model.exe --accelerator=npu ^
    --compiler_plugin_library_dir=C:\Workspace\junwei\LiteRT\bazel-bin\litert\vendors\intel_openvino\compiler ^
    --dispatch_library_dir=C:\Workspace\junwei\LiteRT\bazel-bin\litert\vendors\intel_openvino\dispatch  ^
    --graph=C:\Workspace\junwei\testing\testdata\one_mul.tflite ^
    --print_tensors=true ^
    --compare_numerical=true

> .\bazel-bin\litert\tools\run_model.exe --accelerator=cpu ^
    --graph=C:\Workspace\junwei\testing\testdata\one_mul.tflite ^
    --print_tensors=true ^
    --compare_numerical=true

1, 
bazelisk-windows-amd64 build -c opt //litert/tools:apply_plugin_main
set OPENVINO_NATIVE_DIR=C:\Program Files (x86)\Intel\openvino_2025.3

bazelisk-windows-amd64 build -c opt //litert/vendors/intel_openvino/dispatch:LiteRtDispatch --config=windows 

bazelisk-windows-amd64 build -c opt //litert/vendors/intel_openvino/compiler:LiteRtCompilerPlugin --config=windows 

2, 
cp .\bazel-bin\litert\vendors\intel_openvino\compiler\LiteRtCompilerPlugin.dll .\openvino_prebuilts
cp .\bazel-bin\litert\vendors\intel_openvino\dispatch\LiteRtDispatch.dll .\openvino_prebuilts

cp .\bazel-bin\litert\vendors\intel_openvino\compiler\LiteRtCompilerPlugin.dll C:\Workspace\junwei\gemma\
cp .\bazel-bin\litert\vendors\intel_openvino\dispatch\LiteRtDispatch.dll C:\Workspace\junwei\gemma\

3, Setup OpenVINO enviroment
C:\Workspace\junwei\LiteRT>"C:\Program Files (x86)\Intel\openvino_2025.3\setupvars.bat"
set ZE_INTEL_NPU_PLATFORM_OVERRIDE=LUNARLAKE

4, LiteRT AOT compilation of the prefill_decode model 
.\bazel-bin\litert\tools\apply_plugin_main.exe --cmd apply ^
    --model=C:\Workspace\junwei\gemma\intel_gemma3_epsilon1e4.tflite ^
    -o C:\Workspace\junwei\gemma\intel_gemma3_epsilon1e4_aot.tflite ^
    --libs C:\Workspace\junwei\LiteRT\openvino_prebuilts ^
    --intel_openvino_device_type=npu --soc_manufacturer "IntelOpenVINO"

There are a known issue that the AOT TFLite model can't run on Windows, so the CompilerCache model is a
workaround:
C:\Workspace\junwei\LiteRT>.\bazel-bin\litert\tools\benchmark_model.exe ^
    --graph=C:\Workspace\junwei\gemma\intel_gemma3_epsilon1e4.tflite ^
    --use_npu=true ^
    --compiler_plugin_library_path=C:\Workspace\junwei\LiteRT\bazel-bin\litert\vendors\intel_openvino\compiler ^
    --dispatch_library_path=C:\Workspace\junwei\LiteRT\bazel-bin\litert\vendors\intel_openvino\dispatch ^
    --compiler_cache_path=C:\Workspace\junwei\gemma

5, LiteRT-LM Model compilation
set SOURCE_DIR=C:\Workspace\junwei\gemma\original_list_of_models_for_gemma3
C:\Workspace\junwei\LiteRT-LM>
bazelisk-windows-amd64 --output_base=C:\bzl run //schema/py:litertlm_builder_cli -- ^
  system_metadata --str Authors "ODML team" ^
  sp_tokenizer --path %SOURCE_DIR%\TOKENIZER_MODEL.spiece ^
  llm_metadata --path %SOURCE_DIR%\METADATA.pb ^
  tflite_model --path %SOURCE_DIR%\TF_LITE_EMBEDDER.tflite --model_type embedder  --str_metadata model_version "1.0.1" ^
  tflite_model --path %SOURCE_DIR%\TF_LITE_AUX.tflite --model_type aux ^
  tflite_model --path C:\Workspace\junwei\gemma\intel_gemma3_epsilon1e4_aot.tflite --model_type prefill_decode ^
  output --path C:\Workspace\junwei\gemma\intel_gemma3_epsilon1e4_aot_2.litertlm

6, Latest code for litert_lm_main does not recognize the benchmark* flags. Hence checkout to it's parent commit:
C:\Workspace\junwei\LiteRT-LM>git checkout bc2cf3a
bazelisk-windows-amd64.exe --output_base=C:\bzl build --define=DISABLE_HUGGINGFACE_TOKENIZER=1 --config=windows //runtime/engine:litert_lm_main


7, 
LiteRtCompilerPlugin.dll and LiteRtDispatch.dll must be in the same directory as model.
C:\Workspace\junwei\LiteRT-LM>.\bazel-bin\runtime\engine\litert_lm_main.exe ^
    --backend=npu --model_path=C:\Workspace\junwei\gemma\intel_gemma3_epsilon1e4_aot_2.litertlm ^
    --input_prompt="What is the capital of France"

C:\Workspace\junwei\LiteRT-LM>.\bazel-bin\runtime\engine\litert_lm_main.exe ^
   --backend=npu --model_path=C:\Workspace\junwei\gemma\intel_gemma3_epsilon1e4_aot_2.litertlm ^
   --benchmark --benchmark_prefill_tokens=1024 --benchmark_decode_tokens=256 --async=false




===========mobilenet aot model==================
> C:\Workspace\junwei\LiteRT>.\bazel-bin\litert\tools\apply_plugin_main.exe --cmd apply --model=C:\Workspace\junwei\dump_tflite_models\mobilenet_v2_1.0_224.tflite -o C:\Workspace\junwei\dump_tflite_models\mobilenet_aot.tflite --libs C:\Workspace\junwei\LiteRT\openvino_prebuilts --intel_openvino_device_type=npu --soc_manufacturer "IntelOpenVINO"
> .\bazel-bin\litert\tools\run_model.exe --accelerator=npu  --dispatch_library_dir=C:\Workspace\junwei\LiteRT\openvino_prebuilts --graph=C:\Workspace\junwei\dump_tflite_models\mobilenet_aot.tflite


==========debug example plugin with VS=============
C:\Workspace\junwei\LiteRT>bazelisk-windows-amd64 build -c dbg //litert/vendors/examples:LiteRtCompilerPlugin --config=windows
C:\Workspace\junwei\LiteRT>bazelisk-windows-amd64 build -c dbg //litert/tools:apply_plugin_main
C:\Workspace\junwei\LiteRT>.\bazel-bin\litert\tools\apply_plugin_main.exe --cmd apply --model=C:\Workspace\junwei\LiteRT\bazel-bin\litert\test\testdata\one_mul.tflite -o C:\Workspace\junwei\dump_tflite_models\one_mul_aot_example_plugin.tflite --libs C:\Workspace\junwei\LiteRT\bazel-bin\litert\vendors\examples
.\bazel-bin\litert\tools\run_model.exe  --dispatch_library_dir=C:\Workspace\junwei\LiteRT\bazel-bin\litert\vendors\examples --graph=C:\Workspace\junwei\dump_tflite_models\mobilenet_aot_example_plugin.tflite 
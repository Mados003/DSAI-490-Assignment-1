cd "/mnt/e/ZC/YEAR 4 S2/DSAI 490/Assignment 1/DSAI-490-Assignment-1"
source .venv/bin/activate
echo "STEP 1: activate .venv"
echo "STEP 2: versions"
python -V
python -c "import platform, tensorflow as tf; print('python', platform.python_version()); print('tensorflow', tf.__version__)"
echo "STEP 3: site-packages"
SP=$(python -c "import site,os; print(next(p for p in site.getsitepackages() if os.path.isdir(p)))")
echo "site_packages=$SP"
echo "STEP 4: export LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$SP/nvidia/cublas/lib:$SP/nvidia/cuda_cupti/lib:$SP/nvidia/cuda_nvrtc/lib:$SP/nvidia/cuda_runtime/lib:$SP/nvidia/cudnn/lib:$SP/nvidia/cufft/lib:$SP/nvidia/curand/lib:$SP/nvidia/cusolver/lib:$SP/nvidia/cusparse/lib:$SP/nvidia/nccl/lib:$SP/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH}"
echo "export LD_LIBRARY_PATH=\"/usr/lib/wsl/lib:$SP/nvidia/cublas/lib:$SP/nvidia/cuda_cupti/lib:$SP/nvidia/cuda_nvrtc/lib:$SP/nvidia/cuda_runtime/lib:$SP/nvidia/cudnn/lib:$SP/nvidia/cufft/lib:$SP/nvidia/curand/lib:$SP/nvidia/cusolver/lib:$SP/nvidia/cusparse/lib:$SP/nvidia/nccl/lib:$SP/nvidia/nvjitlink/lib:\${LD_LIBRARY_PATH}\""
echo "STEP 5: GPU check"
timeout 120s python -c "import traceback; import tensorflow as tf; print('is_built_with_cuda', tf.test.is_built_with_cuda());\
try:\
 print('physical_gpus', tf.config.list_physical_devices('GPU'))\
except Exception:\
 traceback.print_exc()"
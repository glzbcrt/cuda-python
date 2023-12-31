# cuda-python

cuda-python is a simple Python script ([main.py](./main.py)) that using the [ctypes](https://docs.python.org/3/library/ctypes.html) library will call a function inside a [CUDA DLL file](./kernel.cu). This CUDA file has a very simple kernel to add two vectors using a CUDA enabled GPU.

This is a sample application I created to learn more about:

- [CUDA development](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html).
- [NVIDIA Tools Extension (NVTX)](https://github.com/NVIDIA/NVTX).
- Python and C interoperability.

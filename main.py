from ctypes import Structure, c_double, cdll, POINTER
from random import random


class VectorAddResult(Structure):
    """
    Represents the result of a vector addition operation.

    Attributes:
        amount (float): The sum of the vector elements.
        time (float): The time taken to perform the addition operation.
    """
    _fields_ = [
        ("amount", c_double),
        ("time", c_double)
    ]


# Load the CUDA kernel library.
kernels = cdll.LoadLibrary("""build\Debug\cuda-python.dll""")

# Get the VectorAdd function from the library loaded previously. Note that name mangling changes the function names.
vectorAdd = getattr(kernels, "?VectorAdd@@YAPEAUVECTOR_ADD_RESULT@@I@Z")

# Set the return type of the function to be a pointer to a VectorAddResult structure.
vectorAdd.restype = POINTER(VectorAddResult)

# Call the vectorAdd function with a random number.
result = vectorAdd(c_double(random()))

# Print the result.
print(
    f"""
 sum: {result.contents.amount}
time: {result.contents.time}
""")

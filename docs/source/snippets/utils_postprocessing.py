# [start-torch]
from skrl.utils import postprocessing


# assuming there is a directory called "memories" with Torch files in it
memory_iterator = postprocessing.MemoryFileIterator("memories/*.pt")
for filename, data in memory_iterator:
    filename    # str: basename of the current file
    data    # dict: keys are the names of the memory tensors in the file. 
            # Tensor shapes are (memory size, number of envs, specific content size)
    
    # example of simple usage: print the filenames of all memories and their tensor shapes
    print("\nfilename:", filename)
    print("  |-- states:", data['states'].shape)
    print("  |-- actions:", data['actions'].shape)
    print("  |-- rewards:", data['rewards'].shape)
    print("  |-- next_states:", data['next_states'].shape)
    print("  |-- dones:", data['dones'].shape)
# [end-torch]


# [start-numpy]
from skrl.utils import postprocessing


# assuming there is a directory called "memories" with NumPy files in it
memory_iterator = postprocessing.MemoryFileIterator("memories/*.npz")
for filename, data in memory_iterator:
    filename    # str: basename of the current file
    data    # dict: keys are the names of the memory arrays in the file.
            # Array shapes are (memory size, number of envs, specific content size)
    
    # example of simple usage: print the filenames of all memories and their array shapes
    print("\nfilename:", filename)
    print("  |-- states:", data['states'].shape)
    print("  |-- actions:", data['actions'].shape)
    print("  |-- rewards:", data['rewards'].shape)
    print("  |-- next_states:", data['next_states'].shape)
    print("  |-- dones:", data['dones'].shape)
# [end-numpy]


# [start-csv]
from skrl.utils import postprocessing


# assuming there is a directory called "memories" with CSV files in it
memory_iterator = postprocessing.MemoryFileIterator("memories/*.csv")
for filename, data in memory_iterator:
    filename    # str: basename of the current file
    data    # dict: keys are the names of the memory list of lists extracted from the file.
            # List lengths are (memory size * number of envs) and 
            # sublist lengths are (specific content size)
    
    # example of simple usage: print the filenames of all memories and their list lengths
    print("\nfilename:", filename)
    print("  |-- states:", len(data['states']))
    print("  |-- actions:", len(data['actions']))
    print("  |-- rewards:", len(data['rewards']))
    print("  |-- next_states:", len(data['next_states']))
    print("  |-- dones:", len(data['dones']))
# [end-csv]

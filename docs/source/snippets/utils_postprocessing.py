# [start-memory_file_iterator-torch]
from skrl.utils import postprocessing


# assuming there is a directory called "memories" with Torch files in it
memory_iterator = postprocessing.MemoryFileIterator("memories/*.pt")
for filename, data in memory_iterator:
    filename    # str: basename of the current file
    data    # dict: keys are the names of the memory tensors in the file.
            # Tensor shapes are (memory size, number of envs, specific content size)

    # example of simple usage:
    # print the filenames of all memories and their tensor shapes
    print("\nfilename:", filename)
    print("  |-- states:", data['states'].shape)
    print("  |-- actions:", data['actions'].shape)
    print("  |-- rewards:", data['rewards'].shape)
    print("  |-- next_states:", data['next_states'].shape)
    print("  |-- dones:", data['dones'].shape)
# [end-memory_file_iterator-torch]


# [start-memory_file_iterator-numpy]
from skrl.utils import postprocessing


# assuming there is a directory called "memories" with NumPy files in it
memory_iterator = postprocessing.MemoryFileIterator("memories/*.npz")
for filename, data in memory_iterator:
    filename    # str: basename of the current file
    data    # dict: keys are the names of the memory arrays in the file.
            # Array shapes are (memory size, number of envs, specific content size)

    # example of simple usage:
    # print the filenames of all memories and their array shapes
    print("\nfilename:", filename)
    print("  |-- states:", data['states'].shape)
    print("  |-- actions:", data['actions'].shape)
    print("  |-- rewards:", data['rewards'].shape)
    print("  |-- next_states:", data['next_states'].shape)
    print("  |-- dones:", data['dones'].shape)
# [end-memory_file_iterator-numpy]


# [start-memory_file_iterator-csv]
from skrl.utils import postprocessing


# assuming there is a directory called "memories" with CSV files in it
memory_iterator = postprocessing.MemoryFileIterator("memories/*.csv")
for filename, data in memory_iterator:
    filename    # str: basename of the current file
    data    # dict: keys are the names of the memory list of lists extracted from the file.
            # List lengths are (memory size * number of envs) and
            # sublist lengths are (specific content size)

    # example of simple usage:
    # print the filenames of all memories and their list lengths
    print("\nfilename:", filename)
    print("  |-- states:", len(data['states']))
    print("  |-- actions:", len(data['actions']))
    print("  |-- rewards:", len(data['rewards']))
    print("  |-- next_states:", len(data['next_states']))
    print("  |-- dones:", len(data['dones']))
# [end-memory_file_iterator-csv]


# [start-tensorboard_file_iterator-list]
from skrl.utils import postprocessing


# assuming there is a directory called "runs" with experiments and Tensorboard files in it
tensorboard_iterator = postprocessing.TensorboardFileIterator("runs/*/events.out.tfevents.*", \
    tags=["Reward / Total reward (mean)"])
for dirname, data in tensorboard_iterator:
    dirname    # str: path of the directory (experiment name) containing the Tensorboard file
    data    # dict: keys are the tags, values are lists of [step, value] pairs

    # example of simple usage:
    # print the directory name and the value length for the "Reward / Total reward (mean)" tag
    print("\ndirname:", dirname)
    for tag, values in data.items():
        print("  |-- tag:", tag)
        print("  |   |-- value length:", len(values))
# [end-tensorboard_file_iterator-list]

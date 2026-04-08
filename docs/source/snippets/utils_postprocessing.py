# [start-memory_file_iterator-torch]
from skrl.utils import postprocessing


# assuming there is a directory called "memories" with Torch files in it
memory_iterator = postprocessing.MemoryFileIterator("memories/*.pt")
for filename, data in memory_iterator:
    filename: str  # basename of the current file
    data: dict  # keys: tensor names, values: tensors with shape (memory_size, num_envs, data_size)

    # example of simple usage:
    # print the filenames of all memories and their tensor shapes
    print("\nfilename:", filename)
    print("  |-- observations:", data["observations"].shape)
    print("  |-- actions:", data["actions"].shape)
    print("  |-- rewards:", data["rewards"].shape)
    print("  |-- next_observations:", data["next_observations"].shape)
    print("  |-- terminated:", data["terminated"].shape)
    print("  |-- truncated:", data["truncated"].shape)
# [end-memory_file_iterator-torch]


# [start-memory_file_iterator-numpy]
from skrl.utils import postprocessing


# assuming there is a directory called "memories" with NumPy files in it
memory_iterator = postprocessing.MemoryFileIterator("memories/*.npz")
for filename, data in memory_iterator:
    filename: str  # basename of the current file
    data: dict  # keys: array names, values: arrays with shape (memory_size, num_envs, data_size)

    # example of simple usage:
    # print the filenames of all memories and their array shapes
    print("\nfilename:", filename)
    print("  |-- observations:", data["observations"].shape)
    print("  |-- actions:", data["actions"].shape)
    print("  |-- rewards:", data["rewards"].shape)
    print("  |-- next_observations:", data["next_observations"].shape)
    print("  |-- terminated:", data["terminated"].shape)
    print("  |-- truncated:", data["truncated"].shape)
# [end-memory_file_iterator-numpy]


# [start-memory_file_iterator-csv]
from skrl.utils import postprocessing


# assuming there is a directory called "memories" with CSV files in it
memory_iterator = postprocessing.MemoryFileIterator("memories/*.csv")
for filename, data in memory_iterator:
    filename: str  # basename of the current file
    data: dict  # keys: list names, values: lists with length (memory_size * num_envs) of sub-lists with length (data_size)

    # example of simple usage:
    # print the filenames of all memories and their list lengths
    print("\nfilename:", filename)
    print("  |-- observations:", data["observations"].shape)
    print("  |-- actions:", data["actions"].shape)
    print("  |-- rewards:", data["rewards"].shape)
    print("  |-- next_observations:", data["next_observations"].shape)
    print("  |-- terminated:", data["terminated"].shape)
    print("  |-- truncated:", data["truncated"].shape)
# [end-memory_file_iterator-csv]


# [start-tensorboard_file_iterator-list]
from skrl.utils import postprocessing


# assuming there is a directory called "runs" with experiments and TensorBoard files in it
tensorboard_iterator = postprocessing.TensorboardFileIterator(
    "runs/*/events.out.tfevents.*", tags=["Reward / Total reward (mean)"]
)
for dirname, data in tensorboard_iterator:
    dirname: str  # path of the directory (experiment name) containing the TensorBoard file
    data: dict  # keys: tags, values: lists of [step, value] pairs

    # example of simple usage:
    # print the directory name and the value length for the "Reward / Total reward (mean)" tag
    print("\ndirname:", dirname)
    for tag, values in data.items():
        print("  |-- tag:", tag)
        print("  |   |-- value length:", len(values))
# [end-tensorboard_file_iterator-list]

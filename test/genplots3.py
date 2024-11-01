import glob
import re
import matplotlib.pyplot as plt
import pandas as pd
import seaborn

seaborn.set()

file_pattern = "*/results/stdout/out_*.txt"
stdout_files = glob.glob(file_pattern)

# these are the only relevant ones, at least
patterns = {
    "num_worlds": r"Running with (\d+) Worlds:",
    "allocating": r"\[INFO\] Allocating (\d+) of (\d+) bytes on device",
    "actual_program_time": r"\[BENCHMARK\] Actual program time: ([\d\.]+) ms",
    "avg_simulateKernel_time": r"\[BENCHMARK\] Average individual simulateKernel\(\) time over (\d+) samples: ([\d\.]+) ms",
    "total_kernel_time": r"\[BENCHMARK\] Total time spent in kernel before global convergence: ([\d\.]+) ms",
    "kernel_time_ratio": r"\[BENCHMARK\] Kernel time / total program time: ([\d\.]+)",
}

data = {k:[] for k in patterns}
partitions = {0 : "before", 1 : "after"}

for filename in stdout_files:
    with open(filename, 'r') as f:
        content = f.read()

    # let me put you on game: put any string in a quick cmd -> py -> len("<string>")
    sep = content.split("-" * 27)

    # there is an invalid - * 27 at the top line
    for i, partition in enumerate(sep[1:]):
        for key, pattern in patterns.items():
            match = re.search(pattern, partition)
            if match:
                values = [float(x) if '.' in x else int(x) for x in match.groups()]
                data[partitions[i]][key].append(values)

df = pd.DataFrame(data)
print(df.head())
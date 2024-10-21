import glob
import re
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-darkgrid")

# Initialize dictionaries to store data
worlds_list = []
min_convergence_times = []
max_convergence_times = []
avg_convergence_times = []
avg_time_per_batch_loop = []
total_time_sending_batches = []
avg_individual_simulateKernel_time = []
total_time_in_kernel_before_convergence = []
actual_program_times = []

# Regular expression patterns to extract data
patterns = {
    "worlds": r"Running with (\d+) Worlds:",
    "min_max_avg_convergence": r"\[BENCHMARK\] \(Local\) Min convergence time: ([\d\.]+) ms, Max convergence time: ([\d\.]+) ms, Avg convergence time: ([\d\.]+) ms",
    "avg_time_per_batch_loop": r"\[BENCHMARK\] Average time per batch loop: ([\d\.]+) ms",
    "total_time_sending_batches": r"\[BENCHMARK\] Total time sending and executing batches of simulateKernel\(\): ([\d\.]+) ms",
    "avg_individual_simulateKernel_time": r"\[BENCHMARK\] Average individual simulateKernel\(\) time over \d+ samples: ([\d\.]+) ms",
    "total_time_in_kernel_before_convergence": r"\[BENCHMARK\] Total time spent in kernel before.*: ([\d\.]+) ms",
    "actual_program_time": r"\[BENCHMARK\] Actual program time: ([\d\.]+) ms",
}

file_pattern = "./results/stdout/out_*.txt"
files = glob.glob(file_pattern)
files.sort()

for filename in files:
    with open(filename, "r") as file:
        content = file.read()

        # Extract number of worlds
        worlds_match = re.search(patterns["worlds"], content)
        if worlds_match:
            worlds = int(worlds_match.group(1))
            worlds_list.append(worlds)
        else:
            continue  # Skip if number of worlds is not found

        # Extract min, max, avg convergence times
        convergence_match = re.search(patterns["min_max_avg_convergence"], content)
        if convergence_match:
            min_convergence_times.append(float(convergence_match.group(1)))
            max_convergence_times.append(float(convergence_match.group(2)))
            avg_convergence_times.append(float(convergence_match.group(3)))
        else:
            min_convergence_times.append(None)
            max_convergence_times.append(None)
            avg_convergence_times.append(None)

        # Extract average time per batch loop
        avg_batch_loop_match = re.search(patterns["avg_time_per_batch_loop"], content)
        if avg_batch_loop_match:
            avg_time_per_batch_loop.append(float(avg_batch_loop_match.group(1)))
        else:
            avg_time_per_batch_loop.append(None)

        # Extract total time sending and executing batches
        total_batches_match = re.search(patterns["total_time_sending_batches"], content)
        if total_batches_match:
            total_time_sending_batches.append(float(total_batches_match.group(1)))
        else:
            total_time_sending_batches.append(None)

        # Extract average individual simulateKernel() time
        avg_simulateKernel_match = re.search(patterns["avg_individual_simulateKernel_time"], content)
        if avg_simulateKernel_match:
            avg_individual_simulateKernel_time.append(float(avg_simulateKernel_match.group(1)))
        else:
            avg_individual_simulateKernel_time.append(None)

        # Extract total time in kernel before convergence
        total_kernel_time_match = re.search(patterns["total_time_in_kernel_before_convergence"], content)
        if total_kernel_time_match:
            total_time_in_kernel_before_convergence.append(float(total_kernel_time_match.group(1)))
        else:
            total_time_in_kernel_before_convergence.append(None)

        # Extract actual program time
        actual_program_time_match = re.search(patterns["actual_program_time"], content)
        if actual_program_time_match:
            actual_program_times.append(float(actual_program_time_match.group(1)))
        else:
            actual_program_times.append(None)

# Sort data by number of worlds
sorted_indices = sorted(range(len(worlds_list)), key=lambda k: worlds_list[k])
worlds_list = [worlds_list[i] for i in sorted_indices]
min_convergence_times = [min_convergence_times[i] for i in sorted_indices]
max_convergence_times = [max_convergence_times[i] for i in sorted_indices]
avg_convergence_times = [avg_convergence_times[i] for i in sorted_indices]
avg_time_per_batch_loop = [avg_time_per_batch_loop[i] for i in sorted_indices]
total_time_sending_batches = [total_time_sending_batches[i] for i in sorted_indices]
avg_individual_simulateKernel_time = [avg_individual_simulateKernel_time[i] for i in sorted_indices]
total_time_in_kernel_before_convergence = [total_time_in_kernel_before_convergence[i] for i in sorted_indices]
actual_program_times = [actual_program_times[i] for i in sorted_indices]

# x_position labels should be [2^0, 2^24)
x_positions = range(len(worlds_list))
x_labels = [f"$2^{{{n}}}$" for n in range(len(worlds_list))]

MARKER_SZ = 3
LINE_WIDTH = 2

# Plot 1: Min, Max, Avg convergence times vs Worlds
plt.figure(figsize=(10, 6))
plt.plot(x_positions, min_convergence_times, "o--", label="Min Convergence Time", markersize=MARKER_SZ, linewidth=LINE_WIDTH)
plt.plot(x_positions, max_convergence_times, "s-.", label="Max Convergence Time", markersize=MARKER_SZ, linewidth=LINE_WIDTH)
plt.plot(x_positions, avg_convergence_times, "d-", label="Avg Convergence Time", markersize=MARKER_SZ, linewidth=LINE_WIDTH)
plt.xlabel("Number of Worlds")
plt.yscale("log")
plt.ylabel(r"$\log_{10}(\mathrm{Time}) \, \mathrm{(ms)}$")
plt.title("Local Convergence Times vs Number of Worlds")
plt.xticks(x_positions, x_labels)
plt.legend(frameon=True, framealpha=1.0, edgecolor="gray")
plt.grid(True)
plt.savefig("../docs/figures/convtime_vs_worlds.png", dpi=300)
plt.show(block=False)

# Plot 2: Various times vs Worlds
plt.figure(figsize=(10, 6))
plt.plot(x_positions, avg_time_per_batch_loop, "o--", label="Avg Batch Loop Time", markersize=MARKER_SZ, linewidth=LINE_WIDTH)
plt.plot(x_positions, avg_individual_simulateKernel_time, ":", label="Avg Single simulateKernel() Time", markersize=MARKER_SZ, linewidth=LINE_WIDTH)
plt.xlabel("Number of Worlds")
plt.ylabel("Time (ms)")
plt.title("Loop & Kernel Times vs Number of Worlds")
plt.xticks(x_positions, x_labels)
plt.legend(frameon=True, framealpha=1.0, edgecolor="gray")
plt.grid(True)
plt.savefig("../docs/figures/times_vs_worlds.png", dpi=300)
plt.show(block=False)

# Plot 3: Total Program Time(s) vs Worlds
plt.figure(figsize=(10, 6))
plt.plot(x_positions, total_time_sending_batches, "s--", label="Total Time Sending Batches", markersize=MARKER_SZ, linewidth=LINE_WIDTH)
plt.plot(x_positions, total_time_in_kernel_before_convergence, "d-.", label="Total Time in Kernel Before Convergence", markersize=MARKER_SZ, linewidth=LINE_WIDTH)
plt.plot(x_positions, actual_program_times, "o-", label="Total Program Time", markersize=MARKER_SZ, linewidth=LINE_WIDTH)
plt.xlabel("Number of Worlds")
plt.ylabel(r"$\log_{10}(\mathrm{Time}) \, \mathrm{(ms)}$")
plt.yscale("log")
plt.title("Total Program Time(s) vs Number of Worlds")
plt.xticks(x_positions, x_labels)
plt.legend(frameon=True, framealpha=1.0, edgecolor="gray")
plt.grid(True)
plt.savefig("../docs/figures/actual_time_vs_worlds.png", dpi=300)
plt.show(block=False)

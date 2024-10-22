import glob
import re
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import ticker

# Set seaborn style
sns.set()

# Initialize data structures
pre_num_worlds = []
pre_min_times = []
pre_avg_times = []
pre_max_times = []
pre_avg_batch_loop_times = []
pre_avg_simulate_kernel_times = []

post_num_worlds = []
post_min_times = []
post_avg_times = []
post_max_times = []
post_avg_batch_loop_times = []
post_avg_simulate_kernel_times = []

file_pattern = "./results/stdout/out_*.txt"
file_list = glob.glob(file_pattern)

# Regex patterns to extract required data
pattern_convergence_times = r"\[BENCHMARK\] \(Local\) Min convergence time: ([\d\.]+) ms, Max convergence time: ([\d\.]+) ms, Avg convergence time: ([\d\.]+) ms"
pattern_avg_batch_loop_time = r"\[BENCHMARK\] Average time per batch loop: ([\d\.]+) ms"
pattern_avg_simulate_kernel_time = r"\[BENCHMARK\] Average individual simulateKernel\(\) time over [\d]+ samples: ([\d\.]+) ms"

for filename in file_list:
    with open(filename, "r") as f:
        optimization_state = "pre"
        num_world = None
        for i, line in enumerate(f):
            # Ignore first line, it is always the ---------
            if i == 0:
                continue
            line = line.strip()
            if line == "---------------------------":
                if optimization_state == "pre":
                    optimization_state = "post"
                continue
            elif line.startswith("Running with"):
                # Extract number of worlds
                parts = line.split("Running with")[1].split("Worlds")[0].strip()
                num_world = int(parts)
            else:
                # Match convergence times
                match_conv = re.match(pattern_convergence_times, line)
                if match_conv:
                    min_time = float(match_conv.group(1))
                    max_time = float(match_conv.group(2))
                    avg_time = float(match_conv.group(3))
                    if optimization_state == "pre":
                        pre_num_worlds.append(num_world)
                        pre_min_times.append(min_time)
                        pre_max_times.append(max_time)
                        pre_avg_times.append(avg_time)
                    elif optimization_state == "post":
                        post_num_worlds.append(num_world)
                        post_min_times.append(min_time)
                        post_max_times.append(max_time)
                        post_avg_times.append(avg_time)
                # Match average batch loop time
                match_batch = re.match(pattern_avg_batch_loop_time, line)
                if match_batch:
                    avg_batch_loop_time = float(match_batch.group(1))
                    if optimization_state == "pre":
                        pre_avg_batch_loop_times.append(avg_batch_loop_time)
                    elif optimization_state == "post":
                        post_avg_batch_loop_times.append(avg_batch_loop_time)
                # Match average simulateKernel time
                match_kernel = re.match(pattern_avg_simulate_kernel_time, line)
                if match_kernel:
                    avg_simulate_kernel_time = float(match_kernel.group(1))
                    if optimization_state == "pre":
                        pre_avg_simulate_kernel_times.append(avg_simulate_kernel_time)
                    elif optimization_state == "post":
                        post_avg_simulate_kernel_times.append(avg_simulate_kernel_time)

# Create dataframes and sort data
pre_data = pd.DataFrame({
    "num_worlds": pre_num_worlds,
    "min_times": pre_min_times,
    "avg_times": pre_avg_times,
    "max_times": pre_max_times,
    "avg_batch_loop_times": pre_avg_batch_loop_times,
    "avg_simulate_kernel_times": pre_avg_simulate_kernel_times
}).sort_values("num_worlds")

post_data = pd.DataFrame({
    "num_worlds": post_num_worlds,
    "min_times": post_min_times,
    "avg_times": post_avg_times,
    "max_times": post_max_times,
    "avg_batch_loop_times": post_avg_batch_loop_times,
    "avg_simulate_kernel_times": post_avg_simulate_kernel_times
}).sort_values("num_worlds")

# x_labels = [f"$2^{{{n}}}$" for n in range((worlds_list))]
num_worlds = len(pre_data["num_worlds"])
unique_worlds = sorted(set(pre_data["num_worlds"]).union(set(post_data["num_worlds"])))

x_ticks = unique_worlds
x_labels = [f"$2^{{{n}}}$" for n in range(0, 24)]

# Define colors for the lines
min_color = 'blue'
avg_color = 'orange'
max_color = 'green'

# Plot 1 (log conv time x # Worlds)
plt.figure(figsize=(10, 6))

# Pre-optimization data (solid lines)
plt.plot(pre_data["num_worlds"], pre_data["min_times"], "o-", label="Min Time (pre-optimization)", color=min_color)
plt.plot(pre_data["num_worlds"], pre_data["avg_times"], "o-", label="Avg Time (pre-optimization)", color=avg_color)
plt.plot(pre_data["num_worlds"], pre_data["max_times"], "o-", label="Max Time (pre-optimization)", color=max_color)

# Post-optimization data (dashed lines, same colors)
plt.plot(post_data["num_worlds"], post_data["min_times"], "^--", label="Min Time (post-optimization)", color=min_color)
plt.plot(post_data["num_worlds"], post_data["avg_times"], "^--", label="Avg Time (post-optimization)", color=avg_color)
plt.plot(post_data["num_worlds"], post_data["max_times"], "^--", label="Max Time (post-optimization)", color=max_color)

# plt.xlabel("Number of Worlds")
plt.xlabel(r"$log_2(\mathrm{Number of Worlds})$")
plt.ylabel("Time (ms)")
plt.title("Convergence Times vs Number of Worlds")
plt.xscale("log", base=2)
plt.xticks(x_ticks, x_labels)
plt.legend()

plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0e'))

plt.grid(True)
plt.savefig("../docs/figures/convergence_times_vs_worldsLOG.png")
plt.show()

# Plot 2 (avg times v # worlds)
plt.figure(figsize=(10, 6))

# Pre-optimization data (solid lines)
plt.plot(pre_data["num_worlds"], pre_data["avg_batch_loop_times"], "o-", label="Avg Batch Loop Time (pre-optimization)")
plt.plot(pre_data["num_worlds"], pre_data["avg_simulate_kernel_times"], "o-", label="Avg simulateKernel() Time (pre-optimization)")

# Post-optimization data (dashed lines, same colors)
plt.plot(post_data["num_worlds"], post_data["avg_batch_loop_times"], "^--", label="Avg Batch Loop Time (post-optimization)")
plt.plot(post_data["num_worlds"], post_data["avg_simulate_kernel_times"], "^--", label="Avg simulateKernel() Time (post-optimization)")

plt.xlabel("Number of Worlds")
plt.xticks(x_ticks, x_labels)
plt.ylabel("Time (ms)")
plt.title("Average Times vs Number of Worlds")
plt.legend()
plt.grid(True)
plt.savefig("../docs/figures/avg_times_vs_worlds.png")
plt.show()
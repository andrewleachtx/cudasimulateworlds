import os
import re
import glob
import matplotlib.pyplot as plt

# Assumes you are in project root directory, i.e. `. = CUDASIMULATEWORLDS/`

def parse_output_file(filename):
    num_worlds = None
    min_time = None
    avg_time = None
    max_time = None

    with open(filename, 'r') as f:
        for line in f:
            # Extract the number of worlds
            if 'Batching in' in line and 'worlds' in line:
                # Example: [INFO] Batching in 1024 worlds / 1048575 max blocks
                match = re.search(r'Batching in (\d+) worlds', line)
                if match:
                    num_worlds = int(match.group(1))

            # Extract convergence times
            if '(Local) Min convergence time' in line:
                # Example: [BENCHMARK] (Local) Min convergence time: 107263.000000 ms, Max convergence time: 114111.000000 ms, Avg convergence time: 110814.835938 ms
                match = re.search(r'Min convergence time: ([\d\.]+) ms, Max convergence time: ([\d\.]+) ms, Avg convergence time: ([\d\.]+) ms', line)
                if match:
                    min_time = float(match.group(1))
                    max_time = float(match.group(2))
                    avg_time = float(match.group(3))

    return num_worlds, min_time, avg_time, max_time

def main():
    file_pattern = "./test/results/stdout/output_*.txt"

    files = glob.glob(file_pattern)
    files.sort()

    num_worlds_list = []
    min_times = []
    avg_times = []
    max_times = []

    for filename in files:
        num_worlds, min_time, avg_time, max_time = parse_output_file(filename)
        if num_worlds is not None and min_time is not None:
            num_worlds_list.append(num_worlds)
            min_times.append(min_time)
            avg_times.append(avg_time)
            max_times.append(max_time)
            print(f"Parsed data from {filename}:")
            print(f"  Number of worlds: {num_worlds}")
            print(f"  Min convergence time: {min_time} ms")
            print(f"  Avg convergence time: {avg_time} ms")
            print(f"  Max convergence time: {max_time} ms\n")
        else:
            print(f"Warning: Could not extract data from {filename}")

    # Sort the data by number of worlds
    sorted_data = sorted(zip(num_worlds_list, min_times, avg_times, max_times))
    num_worlds_list, min_times, avg_times, max_times = zip(*sorted_data)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(num_worlds_list, min_times, 'o-', label='Min Convergence Time')
    plt.plot(num_worlds_list, avg_times, 's-', label='Avg Convergence Time')
    plt.plot(num_worlds_list, max_times, '^-', label='Max Convergence Time')

    plt.xlabel('Number of Worlds')
    plt.ylabel('Convergence Time (ms)')
    plt.title('Convergence Times vs. Number of Worlds')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the plot to a file
    plt.savefig(f"./docs/figures/{convergence_times}.png")
    print("Plot saved to docs/figures/'convergence_times.png'")

    # Show the plot
    plt.show()

if __name__ == '__main__':
    main()

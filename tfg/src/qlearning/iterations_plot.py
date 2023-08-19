import csv
import matplotlib.pyplot as plt

# Read data from the CSV
episodes = []
iterations = []

with open('data.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        episodes.append(int(row['episode num']))
        iterations.append(int(row['iteration counter']))

# Plot
plt.figure(figsize=(10,6))
plt.plot(episodes, iterations, '-o', label="Iterations per episode")
plt.title("Number of iterations per episode")
plt.xlabel("Episode Number")
plt.ylabel("Number of Iterations")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

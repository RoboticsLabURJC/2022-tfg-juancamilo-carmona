import csv
import matplotlib.pyplot as plt

# Leer el archivo CSV y obtener los valores de giros
file_name = '/home/alumnos/camilo/Escritorio/qlearning_metrics/actions_1.csv'
steer_values = []

with open(file_name, 'r') as csv_file:
    reader = csv.reader(csv_file)
    for row in reader:
        steer_values.append(float(row[0]))

# Mostrar el histograma
plt.hist(steer_values, bins=21, color='blue', edgecolor='black')
plt.xlabel('Steer Values')
plt.ylabel('Frequency')
plt.title('Histogram of Steer Values')
plt.grid(True)
plt.tight_layout()
plt.show()

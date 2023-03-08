import csv
import matplotlib.pyplot as plt
import numpy as np
import sys

# Abrir archivo csv
column = sys.argv[1]

if column == 'fps':
    column_num = 1

elif column == 'cpu_usage':
    column_num = 2

elif column == 'memory_usage':
    column_num = 3

elif column == 'pid_curling':
    column_num = 4

elif column == 'pid_intensity':
    column_num = 5



data = ['time','fps','cpu usage','Memory usage','PID curling','PID adjustment intesity']

with open('hsv_metrics.csv', 'r') as archivo:
    lector_csv = csv.reader(archivo)

    # Leer datos
    x = []
    y = []

    i = 0
    for fila in lector_csv:

        if i != 0:
            x.append(float(fila[0]))
            y.append(float(fila[int(column_num)]))

        i = i + 1
        
plot_x = np.array(x)
plot_y = np.array(y)

plt.plot(plot_x, plot_y, "b-", label=data[ int(column_num)] )

plt.xlabel('time')
plt.ylabel( data[int(column_num)] )
plt.title(data[int(column_num)]+' plot')
plt.legend(loc="upper left")

plt.show()


    
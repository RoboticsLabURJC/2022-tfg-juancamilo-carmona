import pandas as pd
import matplotlib.pyplot as plt

# Carga los datos del archivo CSV
#df = pd.read_csv('/home/camilo/Escritorio/qlearning_metrics/metrics_1.csv')
df = pd.read_csv('/home/alumnos/Escritorio/qlearning_metrics/metrics_1.csv')

# Grafica el número de episodio en el eje x y la recompensa acumulada en el eje y
plt.plot(df['num episodio'], df['acumulated reward'])
plt.xlabel('Número de episodio')
plt.ylabel('Recompensa acumulada')
plt.title('Número de episodio vs Recompensa acumulada')
plt.show()

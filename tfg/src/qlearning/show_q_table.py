import pickle
import pandas as pd

def cargar_tabla_q(ruta_archivo):
    with open(ruta_archivo, 'rb') as archivo:
        tabla_q = pickle.load(archivo)
    
    for i in range(tabla_q.shape[0]):
        for j in range(tabla_q.shape[1]):
            print(f"Subtabla para dim1={i}, dim2={j}:")
            subtabla = tabla_q[i, j, :, :]
            df = pd.DataFrame(subtabla)
            print(df)
            print("\n" + "="*50 + "\n")

# Uso de la funci�n
ruta = "/home/alumnos/camilo/2022-tfg-juancamilo-carmona/tfg/src/qlearning/q_table.pkl"  # Reemplaza esto con la ruta de tu archivo
cargar_tabla_q(ruta)

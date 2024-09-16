# Regresión de Ancho de Banda con modelos UNET

Este proyecto incluye un conjunto de funciones para procesar matrices, dividir el dataset (train/test/val) y realizar predicciones utilizando un modelo de regresión basado en UNet.

# Requisitos

Asegúrate de tener instaladas las siguientes bibliotecas:

* numpy
* torch
* PIL
* torchvision
* natsort

Para ello, ejecute el siguiente comando:

`pip install numpy torch pillow torchvision natsort`

# Estructura de Directorios

+ `/content/Matrices/` - Directorio principal donde se almacenan las matrices de entrada y salida.
+ `/content/Matrices/IN/` - Subdirectorio para las matrices de entrada.
+ `/content/Matrices/OUT/` - Subdirectorio para las matrices de salida.
+ `/content/Matrices/TEST/IN/` - Subdirectorio para las matrices de entrada de prueba.
+ `/content/Matrices/TEST/OUT/` - Subdirectorio para las matrices de salida de prueba.

# Preprocesado de los datos para generar las matrices de entrada:

- Procesar matrices en base a un directorio:
 
Esta función combina varias matrices en una sola matriz de 5 dimensiones, opcionalmente estandariza algunas matrices y guarda el resultado en formato .npy.

```
import numpy as np
import os
from natsort import natsorted

def procesar_matrices(folder, folder_output, estandarizacion=False):
    # Lista de archivos en la carpeta de deployment
    lista = os.listdir(os.path.join(folder, 'deployment'))
    lista = natsorted(lista)

    for i in range(0, len(lista) - 3):
        deploy_femto = os.path.join(folder, 'deployment', f'{i}_femto')
        deploy_micro = os.path.join(folder, 'deployment', f'{i}_micro')
        deploy_pico = os.path.join(folder, 'deployment', f'{i}_pico')
        
        if os.path.isfile(deploy_femto) and os.path.isfile(deploy_micro) and os.path.isfile(deploy_pico):
            snr_femto = os.path.join(folder, 'snr', f'{i}_femto')
            snr_micro = os.path.join(folder, 'snr', f'{i}_micro')
            snr_pico = os.path.join(folder, 'snr', f'{i}_pico')
            
            if os.path.isfile(snr_femto) and os.path.isfile(snr_micro) and os.path.isfile(snr_pico):
                user_data = os.path.join(folder, 'users')
                pathloss_data = os.path.join(folder, 'pathloss')

                matriz_femto_deploy = np.loadtxt(deploy_femto, dtype=float)
                matriz_micro_deploy = np.loadtxt(deploy_micro, dtype=float)
                matriz_pico_deploy = np.loadtxt(deploy_pico, dtype=float)
                matriz_pathloss = np.loadtxt(pathloss_data, dtype=float)
                matriz_users = np.loadtxt(user_data, dtype=float)

                # Combina las matrices en una sola matriz de 8 dimensiones
                matriz_combinada = np.array([matriz_femto_deploy, matriz_micro_deploy, matriz_pico_deploy,
                                             matriz_pathloss, matriz_users])

                if estandarizacion:
                    # Índices de las matrices que deseas estandarizar
                    indices_estandarizar = [3]

                    # Calcula la media y la desviación estándar para las matrices seleccionadas
                    medias = np.mean(matriz_combinada[indices_estandarizar], axis=(0, 1))
                    desviaciones_estandar = np.std(matriz_combinada[indices_estandarizar], axis=(0, 1))

                    # Estandariza las matrices seleccionadas
                    for j in indices_estandarizar:
                        if desviaciones_estandar[j] != 0:
                            matriz_combinada[j] = (matriz_combinada[j] - medias) / desviaciones_estandar

                ruta_guardado_binario_in = os.path.join(folder_output, 'IN', f'matriz_combinada{i}.npy')
                np.save(ruta_guardado_binario_in, matriz_combinada)

                # Leer los datos desde el archivo CSV para generar los ficheros de salida con el que comparar las soluciones
                data = np.genfromtxt(os.path.join(folder, 'usersInfo', f'{i}.csv'), delimiter=',', skip_header=1)

                # Obtener las coordenadas x, y y capacidad
                x = data[:, 1].astype(int)
                y = data[:, 2].astype(int)
                capacity = data[:, 4]

                # Obtener las dimensiones de la matriz
                max_x, max_y = np.max(x), np.max(y)

                # Crear una matriz de ceros con las dimensiones máximas
                matrix = np.zeros((100, 100))

                # Asignar los valores de capacidad en las posiciones correspondientes
                matrix[x, y] = capacity

                # Guardar la matriz en formato binario de NumPy (npy)
                ruta_guardado_binario_out = os.path.join(folder_output, 'OUT', f'matriz_combinada{i}.npy')
                np.save(ruta_guardado_binario_out, matrix)

# Ejemplo de uso de la función:
procesar_matrices('/content/UNET5G/SAMPLE/adaptive_0/0-0-4605876/ia/raw/', '/content/Matrices/', estandarizacion=False)

```

- Procesar matrices en base a un ejemplo en concreto:

En el caso en el que solo deseemos generar las muestras para un caso en concreto, es necesario ejecutar la siguiente función:

```
import numpy as np

def procesar_matrices(matriz_femto_deploy, matriz_micro_deploy, matriz_pico_deploy,
                      matriz_pathloss, matriz_users, data_users_info,
                      folder_output, estandarizacion=False):
    
    # Combina las matrices en una sola matriz de 8 dimensiones
    matriz_combinada = np.array([matriz_femto_deploy, matriz_micro_deploy, matriz_pico_deploy,
                                 matriz_pathloss, matriz_users])

    if estandarizacion:
        # Índices de las matrices que deseas estandarizar
        indices_estandarizar = [3]

        # Calcula la media y la desviación estándar para las matrices seleccionadas
        medias = np.mean(matriz_combinada[indices_estandarizar], axis=(0, 1))
        desviaciones_estandar = np.std(matriz_combinada[indices_estandarizar], axis=(0, 1))

        # Estandariza las matrices seleccionadas
        for j in indices_estandarizar:
            if desviaciones_estandar[j] != 0:
                matriz_combinada[j] = (matriz_combinada[j] - medias) / desviaciones_estandar

    ruta_guardado_binario_in = os.path.join(folder_output, 'matriz_combinada_ENTRADA.npy')
    np.save(ruta_guardado_binario_in, matriz_combinada)

    # Leer los datos desde el archivo CSV para generar los ficheros de salida con el que comparar las soluciones
    data = np.array(data_users_info)

    # Obtener las coordenadas x, y y capacidad
    x = data[:, 1].astype(int)
    y = data[:, 2].astype(int)
    capacity = data[:, 4]

    # Obtener las dimensiones de la matriz
    max_x, max_y = np.max(x), np.max(y)

    # Crear una matriz de ceros con las dimensiones máximas
    matrix = np.zeros((100, 100))

    # Asignar los valores de capacidad en las posiciones correspondientes
    matrix[x, y] = capacity

    # Guardar la matriz en formato binario de NumPy (npy)
    ruta_guardado_binario_out = os.path.join(folder_output, 'matriz_combinada_SALIDA.npy')
    np.save(ruta_guardado_binario_out, matrix)

# Ejemplo de uso de la función:
# Supongamos que ya tienes las matrices y los datos de usersInfo cargados en las variables correspondientes
# Aquí se están usando valores de ejemplo
matriz_femto_snr = np.loadtxt('/content/UNET5G/SAMPLE/adaptive_0/0-0-4605876/ia/raw/snr/0_femto')
matriz_micro_snr = np.loadtxt('/content/UNET5G/SAMPLE/adaptive_0/0-0-4605876/ia/raw/snr/0_micro')
matriz_pico_snr = np.loadtxt('/content/UNET5G/SAMPLE/adaptive_0/0-0-4605876/ia/raw/snr/0_pico')
matriz_femto_deploy = np.loadtxt('/content/UNET5G/SAMPLE/adaptive_0/0-0-4605876/ia/raw/deployment/0_femto')
matriz_micro_deploy = np.loadtxt('/content/UNET5G/SAMPLE/adaptive_0/0-0-4605876/ia/raw/deployment/0_micro')
matriz_pico_deploy = np.loadtxt('/content/UNET5G/SAMPLE/adaptive_0/0-0-4605876/ia/raw/deployment/0_pico')
matriz_pathloss = np.loadtxt('/content/UNET5G/SAMPLE/adaptive_0/0-0-4605876/ia/raw/pathloss')
matriz_users = np.loadtxt('/content/UNET5G/SAMPLE/adaptive_0/0-0-4605876/ia/raw/users')
data_users_info = np.genfromtxt('/content/UNET5G/SAMPLE/adaptive_0/0-0-4605876/ia/raw/usersInfo/0.csv',delimiter=',', skip_header=1)
 
procesar_matrices(matriz_femto_deploy, matriz_micro_deploy, matriz_pico_deploy,
                  matriz_pathloss, matriz_users, data_users_info,
                  '/content/', estandarizacion=False)
```

- Destinar archivos para test:

Una vez generados las matrices para un directorio en concreto, se destina un porcentaje definido como argumento para test:

```
import os
import random
import shutil

def mover_muestras_para_test(folder_output, porcentaje_test=0.10):
    # Ruta de los archivos de entrada y la ruta de destino para TEST
    ruta_in = os.path.join(folder_output, 'IN')
    ruta_test_in = os.path.join(folder_output, 'TEST', 'IN')
    ruta_test_out = os.path.join(folder_output, 'TEST', 'OUT')
    ruta_out = os.path.join(folder_output, 'OUT')

    # Crear directorios de destino si no existen
    os.makedirs(ruta_test_in, exist_ok=True)
    os.makedirs(ruta_test_out, exist_ok=True)

    # Lista de archivos en la carpeta de entrada
    lista_ficheros = os.listdir(ruta_in)
    
    # Mezclar la lista de archivos aleatoriamente
    random.shuffle(lista_ficheros)
    random.shuffle(lista_ficheros)

    # Calcular el número de archivos para validación
    num_archivos_validacion = int(len(lista_ficheros) * porcentaje_test)
    archivos_validacion_completo = random.sample(lista_ficheros, num_archivos_validacion)

    # Mover los archivos seleccionados a la carpeta de TEST
    for archivo in archivos_validacion_completo:
        origenin = os.path.join(ruta_in, archivo)
        destinoin = os.path.join(ruta_test_in, archivo)
        shutil.move(origenin, destinoin)

        origenout = os.path.join(ruta_out, archivo)
        destinoout = os.path.join(ruta_test_out, archivo)
        shutil.move(origenout, destinoout)

    # Imprimir algunos detalles para verificación
    if len(lista_ficheros) > 2 and len(archivos_validacion_completo) > 2:
        print(lista_ficheros[1], lista_ficheros[2])
        print(len(archivos_validacion_completo))
        print(archivos_validacion_completo[1], archivos_validacion_completo[2])
```

# GPU - Realizar regresión con el modelo entrenado previamente:

Estas funciones cargan un modelo desde un checkpoint y realiza predicciones sobre un conjunto de archivos de entrada.

```
import argparse
import logging
import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from utils.data_loading import CustomDataset 
from unet import UNet 
from torch.nn import DataParallel

def predict_regression(net, input_data, device):
    net.eval()
    input_data = torch.from_numpy(input_data).unsqueeze(0).float()
    #input_data = input_data.unsqueeze(0)
    input_data = input_data.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        output = net(input_data).cpu()
        regression_values = output.squeeze(0).numpy()
    return regression_values

def load_model(model):
    net = torch.nn.DataParallel(UNet(n_channels=5, n_classes=3, bilinear=False))  # Update with your model definition
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    state_dict = torch.load(model, map_location=device)
    net.load_state_dict(state_dict)
    return (net,device)

def make_pred(net_pack,input_data):
    input_matrix = np.load(input_data)
    regression_values = predict_regression(net=net_pack[0],input_data=input_matrix,device=net_pack[1])
    
```

# CPU - En caso de que se quiera inferir usando CPU, el único cambio a realizar será la forma en la que se carga el modelo:

```
def load_model_cpu(model):
    # Cargar el modelo sin usar DataParallel
    net = UNet(n_channels=5, n_classes=3, bilinear=False)
    device = torch.device('cpu')  # Usar CPU
    net.to(device=device)

    # Cargar el estado del modelo y corregir el prefijo 'module.'
    state_dict = torch.load(model, map_location=device)
    new_state_dict = {}
    for key in state_dict:
        new_key = key.replace('module.', '')  # Eliminar el prefijo 'module.'
        new_state_dict[new_key] = state_dict[key]

    net.load_state_dict(new_state_dict)
    return (net, device)
```

Ejemplo de uso:

Posicionados en el siguiente directorio:

`./UNET5G/Code/Pytorch-UNet/`

Es necesario dar como argumento la ruta donde se encuentra el modelo, la matriz de dimensionalidad 8 como entrada, así como la ruta donde guardar la matriz predicha antes de ejecutar la correspondiente función.

```
net = load_model('/opt/share/MERIDA/Code/Pytorch-UNet/MODEL.pth')
make_pred(net,'/opt/share/MERIDA/DATASET-TESTALL3/TEST/IN/matriz_combinada4315.npy')
```

Se ha subido una versión del modelo previamente entrenado en el siguiente enlace:

https://mega.nz/file/bZxAmKwA#g7Ize6XZf-O_XsDEthb_LDUh7sO4Pz-NiRL8DFRsUhw


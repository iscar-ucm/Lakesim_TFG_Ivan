# Lakesim_TFG_Ivan
SIMULADORES DE DINÁMICA DE FLUIDOS MEDIANTE LATTICE-BOLTZMANN (LB) Y EULER (EF) SENCILLO CON GRAVEDAD.

Cada carpeta contiene:
  1. Un archivo .py con el algoritmo implementado en Python para simular el fluido en cada caso. Este programa genera: un archivo .txt con los tiempos de computación de cada iteración del método y dos archivos .json con los valores de la velocidad y la presión/densidad de ciertas iteraciones.
  2. Un archivo Plots().py que accede a los archivos de velocidad y presión/densidad generados en el simulador y devuelve un archivo .png con el plot de esos valores por cada archivo .json existente. Además, una vez generado todos los plots, este programa compila todos los plots de velocidad y todos los plots de presión/densidad en sendos archivos de vídeo .avi.
  3. Dos archivos de vídeo .avi, uno con los plots de velocidad y otro con los plots de presión/densidad de una simulación realizada previamente.
  4. La carpeta EMBALSE contiene, además de todo lo anterior, un archivo .json con la malla empleada para simular la geometría de la Laguna del Campillo. Este archivo es necesario para poder ejecutar el simulador.

Al margen de las carpetas con los simuladores, se incluye un anexo complementario con desarrollos matemáticos de expresiones empleadas en el TFG pero que en NINGÚN CASO es necesario para poder seguir el contenido del trabajo. 

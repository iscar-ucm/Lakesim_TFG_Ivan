import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm
import os
import json
import cv2
import shutil

#Crear directorios
folder_padre = os.path.dirname(os.path.abspath(__file__))
folder_densidad = folder_padre + '/Densidades'
folder_velocidad = folder_padre + '/Velocidades'

folder_plots_densidad = folder_padre + '/Plots_densidad'
if os.path.exists(folder_plots_densidad):
    shutil.rmtree(folder_plots_densidad)
os.makedirs(folder_plots_densidad)

folder_plots_velocidad = folder_padre + '/Plots_velocidad'
if os.path.exists(folder_plots_velocidad):
    shutil.rmtree(folder_plots_velocidad)
os.makedirs(folder_plots_velocidad)

#Parámetros simulación
numero_frames = 10_000
fps = 10

with open('Malla Campillo.json', 'r') as archivo:
    malla_json = json.load(archivo)

malla_CAMPILLO = np.array(malla_json)
dimensiones = malla_CAMPILLO.shape

n_x = dimensiones[0]
n_y = dimensiones[1]
n_z = dimensiones[2]

x = np.arange(n_x)
y = np.arange(n_y)
z = np.arange(n_z)
X, Y, Z = np.meshgrid(x, y, z, indexing = "ij")

l_real = 1 #metros
t_real = np.sqrt(3) * l_real #paso de tiempo de cada iteración (s)

def escalar_ejes(valor, _):
    return f"{valor * l_real:.1f}"

obstacle = ~ malla_CAMPILLO.astype(bool)
obstacle[-6:0, 135:149, 6:8] = 0
obstacle[0:2, 11:25, 6:8] = 0

z_planta = 6
x_obs_planta, y_obs_planta = np.where(obstacle[:, :, z_planta] == 1)

y_alzado = 80
x_obs_alzado, z_obs_alzado = np.where(obstacle[:, y_alzado, :] == 1)

inflow_vel = 0.05

#Streamlines
def get_vel_field(x, y, z, macro_vel, field):
    i_0 = int(x)
    j_0 = int(y)
    k_0 = int(z)

    tx = x - i_0
    ty = y - j_0
    tz = z - k_0

    sx = 1 - tx
    sy = 1 - ty
    sz = 1 - tz

    value = (sx*sy*sz*macro_vel[i_0, j_0, k_0, field] + tx*sy*sz*macro_vel[i_0 + 1 , j_0, k_0, field] 
            + sx*ty*sz*macro_vel[i_0, j_0 + 1, k_0, field] + tx*ty*sz*macro_vel[i_0 + 1, j_0 + 1, k_0, field]
            + sx*sy*tz*macro_vel[i_0, j_0, k_0 + 1, field] + tx*sy*tz*macro_vel[i_0 + 1, j_0, k_0 + 1, field]
            + sx*ty*tz*macro_vel[i_0, j_0 + 1, k_0 +1, field] + tx*ty*tz*macro_vel[i_0 + 1, j_0 + 1, k_0 +1, field])
         
    return value


def get_streamlines(i, j, k, macro_vel):
    seg_L = 0.2
    seg_Num = 50

    x_0 = i
    y_0 = j
    z_0 = k

    x = [x_0]
    y = [y_0]
    z = [z_0]

    for n in range(seg_Num):
        if (x_0 >= n_x-1) or (x_0 <= 0) or (y_0 >= n_y-1) or (y_0 <= 0) or (z_0 >= n_z-1) or (z_0 <= 0):
            break 

        u = get_vel_field(x_0, y_0, z_0, macro_vel, 0)
        v = get_vel_field(x_0, y_0, z_0, macro_vel, 1)
        w = get_vel_field(x_0, y_0, z_0, macro_vel, -1)

        l = np.sqrt(u**2 + v**2 + w**2)

        if l==0:
            break

        x_0 += u * seg_L / l
        y_0 += v * seg_L / l
        z_0 += w * seg_L / l

        x = np.append(x, x_0)
        y = np.append(y, y_0)
        z = np.append(z, z_0)

    return x, y, z

for dat in tqdm(range(numero_frames)):
    dat += 1
    archivo_densidad = os.path.join(folder_densidad, 'Densidades_' + str(dat) + '(CampilloLB3D).json')
    archivo_velocidad = os.path.join(folder_velocidad, 'Velocidades_' + str(dat) + '(CampilloLB3D).json')

    if os.path.exists(archivo_densidad) == True:
        with open(archivo_densidad, 'r') as archivo:
            density_json = json.load(archivo)
        density = np.array(density_json)

        with open(archivo_velocidad, 'r') as archivo:
            macro_vel_json = json.load(archivo)
        macro_vel = np.array(macro_vel_json)
        vel_magnitude = np.linalg.norm(macro_vel, axis=-1, ord=2)

        #Plot_Densidad
        fig, axs = plt.subplots(2, 1, sharex= 'row', figsize= (8, 5)) 
        
        iter = 100 * (dat - 1)
        t_iter = f"{iter * t_real / 60:.2f}"
        fig.suptitle('t = ' + str(t_iter) + ' min', fontsize= 20)
    
            #Plot planta
        density_planta = density[:, :, z_planta].T

        im1 = axs[0].imshow(density_planta, origin= 'lower', cmap= 'rainbow', vmin=1, vmax=2)
            
        #axs[0].set_title('Vista de Planta (Z = %i)' %(z_planta), loc='center', fontsize= 10)

        axs[0].scatter(x_obs_planta, y_obs_planta, s= 10, marker='s', color='black')
        axs[0].text(15, 220, 'Planta (Z = %i)' %(z_planta), fontsize = 8,
                bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='lightgray'))


        for i in range(1, n_x, 10):
            for j in range(1, n_y, 10):
                if obstacle[i, j, z_planta] == False:
                    x, y, z = get_streamlines(i, j, z_planta, macro_vel)
                    axs[0].plot(x, y, color= 'black', linewidth = 0.5)

            #Plot alzado
        density_alzado = density[:, y_alzado, :].T
            
        im2 = axs[1].imshow(density_alzado, origin= 'lower', cmap= 'rainbow', vmin=1, vmax=2)
        #axs[1].set_title('Vista de Alzado (Y = %i)' %(y_alzado), loc='center', fontsize= 10)
            
        axs[1].scatter(x_obs_alzado, z_obs_alzado, s= 10, marker='s', color='black')
        axs[1].text(15, 2, 'Alzado (Y = %i)' %(y_alzado), fontsize = 8,
        bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='lightgray'))

        for i in range(1, n_x, 8):
                for k in range(1, n_z, 2):
                    if obstacle[i, y_alzado, k] == False:
                        x, y, z = get_streamlines(i, y_alzado, k, macro_vel)
                        axs[1].plot(x, z, color= 'black', linewidth = 0.5)

        axs[0].xaxis.set_visible(False)
        axs[0].yaxis.set_major_formatter(plt.FuncFormatter(escalar_ejes))

        axs[1].xaxis.set_major_formatter(plt.FuncFormatter(escalar_ejes))
        axs[1].yaxis.set_major_formatter(plt.FuncFormatter(escalar_ejes))

        axs[1].set_aspect(7)

        entrada_y = np.arange(11, 25)
        entrada_x = np.ones(len(entrada_y))
        axs[0].scatter(entrada_x, entrada_y, marker='s', color='lime')

        salida_y = np.arange(135, 149) 
        salida_x = (n_x - 2) * np.ones(len(salida_y))
        axs[0].scatter(salida_x, salida_y, marker='s', color='red')

        cbar = fig.colorbar(im2, ax=axs.ravel().tolist())
        cbar.set_label('Densidad', fontsize= 15)
        cbar.ax.tick_params(labelsize=12)

        frame_iter = "{:06d}".format(iter)
            
        frame_name_densidad = os.path.join(folder_plots_densidad, 'Frame_' + str(frame_iter))
        fig.savefig(frame_name_densidad)
        plt.clf()
        plt.close()

        #Plot Velocidad
        fig, axs = plt.subplots(2, 1, sharex= 'row', figsize= (8, 5)) 
        
        fig.suptitle('t = ' + str(t_iter) + ' min', fontsize= 20)
    
            #Plot planta
        vel_magnitude_planta = vel_magnitude[:, :, z_planta].T

        im1 = axs[0].imshow(vel_magnitude_planta, origin= 'lower', cmap= 'rainbow', vmin=0, vmax= 2*inflow_vel)
        #axs[0].set_title('Vista de Planta (Z = %i)' %(z_planta), loc='center', fontsize= 10)

        axs[0].scatter(x_obs_planta, y_obs_planta, s= 10, marker='s', color='black')
        axs[0].text(15, 220, 'Planta (Z = %i)' %(z_planta), fontsize = 8,
                bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='lightgray'))

        for i in range(1, n_x, 10):
                for j in range(1, n_y, 10):
                    if obstacle[i, j, z_planta] == False:
                        x, y, z = get_streamlines(i, j, z_planta, macro_vel)
                        axs[0].plot(x, y, color= 'black', linewidth = 0.5)

            #Plot alzado
        vel_magnitude_alzado = vel_magnitude[:, y_alzado, :].T
            
        im2 = axs[1].imshow(vel_magnitude_alzado, origin= 'lower', cmap= 'rainbow', vmin=0, vmax= 2*inflow_vel)
        #axs[1].set_title('Vista de Alzado (Y = %i)' %(y_alzado), loc='center', fontsize= 10)
            
        axs[1].scatter(x_obs_alzado, z_obs_alzado, s= 10, marker='s', color='black')
        axs[1].text(15, 2, 'Alzado (Y = %i)' %(y_alzado), fontsize = 8,
                bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='lightgray'))

        for i in range(1, n_x, 8):
                for k in range(1, n_z, 2):
                    if obstacle[i, y_alzado, k] == False:
                        x, y, z = get_streamlines(i, y_alzado, k, macro_vel)
                        axs[1].plot(x, z, color= 'black', linewidth = 0.5)

        axs[0].xaxis.set_visible(False)
        axs[0].yaxis.set_major_formatter(plt.FuncFormatter(escalar_ejes))

        axs[1].xaxis.set_major_formatter(plt.FuncFormatter(escalar_ejes))
        axs[1].yaxis.set_major_formatter(plt.FuncFormatter(escalar_ejes))

        axs[1].set_aspect(7)

        entrada_y = np.arange(11, 25)
        entrada_x = np.ones(len(entrada_y))
        axs[0].scatter(entrada_x, entrada_y, marker='s', color='lime')

        salida_y = np.arange(135, 149) 
        salida_x = (n_x - 2) * np.ones(len(salida_y))
        axs[0].scatter(salida_x, salida_y, marker='s', color='red')

        cbar = fig.colorbar(im1, ax=axs.ravel().tolist())
        cbar.set_label('Módulo del vector velocidad', fontsize= 15)
        cbar.ax.tick_params(labelsize=12)

        frame_iter = "{:06d}".format(iter)
            
        frame_name_velocidad = os.path.join(folder_plots_velocidad, 'Frame_' + str(frame_iter))
        fig.savefig(frame_name_velocidad)
        plt.clf()
        plt.close()

def create_video_from_images(image_folder, video_name, fps):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

#Compilador vídeo Densidad
video_name = folder_padre + '/Campillo_LB3D_densidad.avi'
create_video_from_images(folder_plots_densidad, video_name, fps)

#Compilador vídeo Velocidad
video_name = folder_padre + '/Campillo_LB3D_velocidad.avi'
create_video_from_images(folder_plots_velocidad, video_name, fps)

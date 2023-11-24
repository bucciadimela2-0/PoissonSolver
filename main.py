import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Importa le classi personalizzate
from MultigridSolver2 import MultigridSolver2
from PoissonDiscretization2D import PoissonDiscretization2D
from Statistics import Statistics

if __name__ == "__main__":

    # Parametri del problema
    Nx = 1000
    Ny = 1000
    Lx = 10
    Ly = 10
    x_c = 5
    y_c= 5
    k = 1
    q = 1
    epsilon_0 = 8.854187817e-12  # Costante dielettrica del vuoto

    

    # Inizializza il solver PoissonDiscretization2D
    solver = PoissonDiscretization2D(Nx, Ny, Lx, Ly, epsilon_0, 100, 1e-10)

    
    
    # Inizializza l'oggetto per le statistiche
    Stat = Statistics(Nx,Ny,Lx,Ly,epsilon_0,3,tolerance=1e-10)
    
    Stat.plot_electric_solution()
    #Stat.sample_test()

    #Stat.display_all_functions()

    #Stat.display_all_functions()

    grid_sizes = np.arange(100, 600, 100)
    #Stat.convergence_test(grid_sizes)

     # Esegui il test di complessità
    grid_sizes = np.arange(100, 301, 100)
    #Stat.test_complexity(grid_sizes)

     # Esegui il test esatto
    #error = Stat.test_exact(solver, x_c, y_c, q ,k , Lx,Ly, Nx,Ny)
    #print(f"RMSE: {error}")

    


    # Visualizza le funzioni
    #Stat.display_all_functions()

   

    
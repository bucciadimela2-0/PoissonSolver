import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from MultigridSolver2 import MultigridSolver2
from PoissonDiscretization2D import PoissonDiscretization2D


class Statistics:
    def __init__(self, Nx, Ny, Lx, Ly, epsilon_0, num_iterations, tolerance):
        self.Nx = Nx
        self.Ny = Ny
        self.Lx = Lx
        self.Ly = Ly
        self.x_c= 0
        self.y_c= 0
        self.q= 1
        self.k=1
        self.epsilon_0 = epsilon_0
        self.num_iterations = num_iterations
        self.tolerance = tolerance
        self.iteration = 10
        
        self.solver = PoissonDiscretization2D(Nx,Ny,Lx,Ly,epsilon_0,num_iterations,tolerance)
        self.matrix_solver = MultigridSolver2(Nx,Ny,Lx,Ly, 5,epsilon_0)
        

    def gussian_charge(self, x, y):

        """
        Restituisce una distribuzione di carica gaussiana.
        
        """

        x_center, y_center = self.Lx / 2.0, self.Ly / 2.0
        r = np.sqrt((x - x_center)**2 + (y - y_center)**2)
        return np.exp(-r**2)

    def sin_cos_charge(self, x, y):

        """
        Restituisce una distribuzione di carica sinusoidale e cosinusoidale.

        """

        x_center, y_center = self.Lx / 3.0, self.Ly / 3.0
        r = np.sqrt((x - x_center)**2 + (y - y_center)**2)
        return np.cos(r)

    def radial_exp_charge(self, x, y):
        """
         Restituisce una distribuzione di carica radiale esponenziale.

        """
        x_center, y_center = self.Lx / 2.0, self.Ly / 2.0
        r = np.sqrt((x - x_center)**2 + (y - y_center)**2)
        return np.sin(r)
    
    def uniform_charge(self,x, y):
        """
        Restituisce una distribuzione di carica uniforme.
        """

        return np.ones_like(x)

    def known_potential_uniform_charge(self, x, y):
        """
        Restituisce il potenziale noto associato a una distribuzione di carica uniforme.

        """

        return self.k * self.q / np.sqrt((x - self.x_c)**2 + (y - self.y_c)**2)

    def linear_charge(self, x, y):

        """
            Restituisce una distribuzione di carica lineare.
        """


        return x

    def known_potential_linear_charge(self, x, y):

        """
        Restituisce il potenziale noto associato a una distribuzione di carica lineare.
        """


        return self.k * self.q / np.sqrt((x - self.x_c)**2 + (y - self.y_c)**2)

    def ring_charge(self, x, y):
        """
        Restituisce una distribuzione di carica ad anello.

         """
        x_center, y_center = self.Lx / 2.0, self.Ly / 2.0
        r = np.sqrt((x - x_center)**2 + (y - y_center)**2)
        return np.where(np.logical_and(r >= 20, r <= 30), 1, 0)

    def point_charge(self, x, y):
        """
        Restituisce una distribuzione di carica puntiforme.

        """

        x_center, y_center = self.Lx / 2.0, self.Ly / 2.0
        r = np.sqrt((x - x_center)**2 + (y - y_center)**2)
        return np.where(r == 0, 1, 0)

    def exponential_charge(self, x, y):
        """
        Restituisce una distribuzione di carica esponenziale.

        """

        x_center, y_center = self.Lx / 2.0, self.Ly / 2.0
        r = np.sqrt((x - x_center)**2 + (y - y_center)**2)
        return np.exp(-r**2)

    def spherical_charge(self, x, y):

        """
        Restituisce una distribuzione di carica sferica.

        """
    
        x_center, y_center = self.Lx / 2.0, self.Ly / 2.0
        r = np.sqrt((x - x_center)**2 + (y - y_center)**2)
        return np.where(r < 25, 1, 0)


    def ring_charge(self, x, y):

        """
        Restituisce una distribuzione di carica ad anello.
        """

        x_center, y_center = self.Lx / 2.0, self.Ly / 2.0
        r = np.sqrt((x - x_center)**2 + (y - y_center)**2)
        return np.where(np.logical_and(r >= 20, r <= 30), 1, 0)

    def random_charge_distribution(self, x, y ):

        """
        Restituisce una distribuzione di carica casuale all'interno di un intervallo specifico.
        """

        x_min, x_max = 10, 40  # Intervallo di coordinate x
        y_min, y_max = 10, 40  # Intervallo di coordinate y

        #x_rand = np.random.uniform(x_min, x_max, x.shape)
        #y_rand = np.random.uniform(y_min, y_max, y.shape)

        r = np.sqrt((x - self.x_c)**2 + (y - self.y_c)**2)
        return self.q * np.where(np.logical_and(r >= 10, r <= 20), 1, 0)



    def known_potential_ring_charge(self,x,y):

        """
        Restituisce una il potenziale noto di una distribuzione di carica ad anello.
        """
        return self.k * self.q / np.sqrt((x - self.x_c)**2 + (y - self.y_c)**2)


    def triangular_charge(self,x, y):
        """
        Restituisce una distribuzione di carica triangolare.

        """
        x_center, y_center = self.Lx / 2.0, self.Ly / 2.0
        r = np.sqrt((x - x_center)**2 + (y - y_center)**2)
        return np.where(r < 25, 1 - r / 25, 0)

    
    def surface_charge(self, x, y):
        """
         Restituisce una distribuzione di carica superficiale.
        """
        return np.sin(np.pi * x / self.Lx)

    def known_potential_surface_charge(self, x, y):

        """
        Restituisce il potenziale noto associato a una distribuzione di carica superficiale.
        """
        return self.k * self.q / np.sqrt((x - self.x_c)**2 + (y - self.y_c)**2)


    def delta_charge(self, x, y):
        """
        Restituisce una distribuzione di carica puntiforme delta.

        """
        return self.q * np.where((x == self.x_c) & (y == self.y_c), 1, 0)

    
    def known_potential(self, x, y):

        """
        Restituisce il potenziale noto associato a una distribuzione di carica.
        """

        return self.k * self.q / np.sqrt((x - self.x_c)**2 + (y - self.y_c)**2)
    
    def calculate_error(self, V_known, V_num):
        """
            Calcola l'errore radice dell'errore quadratico medio (RMSE) tra il potenziale noto e il potenziale calcolato.

            Args:
                V_known (numpy.ndarray): Potenziale noto.
                V_num (numpy.ndarray): Potenziale calcolato.

            Returns:
                float: Radice dell'errore quadratico medio (RMSE).
         """

        N = len(V_known)
        rmse = np.sqrt(np.sum((V_known - V_num)**2) / N)
        return rmse


    def sample_test(self):

            methods = [self.solver.solve_gauss_seidel, self.solver.solve_jacobi, lambda grid: self.solver.solve_sor(grid, 1.8), lambda grid: self.matrix_solver.solve_poisson_equation_2d(10,5)]

    
            
        # Set the current rho function
            self.solver.set_rho_function(self.sin_cos_charge)
            self.matrix_solver.set_rho_function(self.sin_cos_charge)

            # Discretize the Poisson equation to get the reference solution
            phi_solution = self.solver.discretize()
            fig, axs = plt.subplots(1, len(methods), figsize=(6*len(methods), 6), sharex=True, sharey=True)

            for j, method in enumerate(methods):
                
                grid = np.zeros((self.Nx, self.Ny))
                grid = method(grid)

                # Crea una griglia di coordinate x e y
                x = np.linspace(0, self.Lx, self.Nx)
                y = np.linspace(0, self.Ly, self.Ny)
                X, Y = np.meshgrid(x, y)

                ax = axs[j]

                # Traccia le linee di livello del potenziale con una mappa di colori personalizzata
                contour = ax.contourf(X, Y, grid, levels=20, cmap='viridis')

                # Aggiungi una barra dei colori con etichette personalizzate
                #cbar = fig.colorbar(contour)
                #cbar.set_label('Potenziale Elettrico')

                # Aggiungi una legenda
                ax.legend([contour.collections[0]], ['Potenziale'], loc='upper right')

                # Aggiungi etichette e titolo
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                

                


            # Aggiungi una barra dei colori con etichette personalizzate
            cbar = fig.colorbar(contour)
            cbar.set_label('Potenziale Elettrico')
            # Aggiungi spaziatura tra i subplots
            plt.tight_layout()

            # Mostra il grafico
            plt.show()



    def display_all_functions(self):

        """
        Mostra i grafici dei potenziali elettrici per diverse distribuzioni di carica e metodi di risoluzione.
        """

        rho_functions = [self.surface_charge, self.delta_charge, self.annular_charge, self.gussian_charge,self.sin_cos_charge,self.radial_exp_charge, self.uniform_charge,self.linear_charge, self.ring_charge,self.point_charge,self.exponential_charge,self.spherical_charge, self.annular_charge, self.triangular_charge]
        methods = [self.solver.solve_gauss_seidel, self.solver.solve_jacobi, lambda grid: self.solver.solve_sor(grid, 1.8), lambda grid: self.matrix_solver.solve_poisson_equation_2d(10,5)]

        

        num_rows = len(rho_functions)
        num_cols = len(methods)

        for i, rho_function in enumerate(rho_functions):
            fig, axs = plt.subplots(1, len(methods), figsize=(6*len(methods), 6), sharex=True, sharey=True)

            # Set the current rho function
            self.solver.set_rho_function(rho_function)
            self.matrix_solver.set_rho_function(rho_function)

            # Discretize the Poisson equation to get the reference solution
            phi_solution = self.solver.discretize()

            for j, method in enumerate(methods):
                grid = np.zeros((self.Nx, self.Ny))
                grid = method(grid)

                # Crea una griglia di coordinate x e y
                x = np.linspace(0, self.Lx, self.Nx)
                y = np.linspace(0, self.Ly, self.Ny)
                X, Y = np.meshgrid(x, y)

                # Crea una figura e un'asse
                ax = axs[j]

                # Traccia le linee di livello del potenziale con una mappa di colori personalizzata
                contour = ax.contourf(X, Y, grid, levels=20, cmap='viridis')

                # Aggiungi una barra dei colori con etichette personalizzate
                #cbar = fig.colorbar(contour)
                #cbar.set_label('Potenziale Elettrico')

                # Aggiungi una legenda
                ax.legend([contour.collections[0]], ['Potenziale'], loc='upper right')

                # Aggiungi etichette e titolo
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_title(f'{rho_function.__name__} - {method.__name__}')

                # Aggiungi griglia
                ax.grid(True, linestyle='--', alpha=0.5)

            # Aggiungi una barra dei colori con etichette personalizzate
            cbar = fig.colorbar(contour)
            cbar.set_label('Potenziale Elettrico')
            # Aggiungi spaziatura tra i subplots
            plt.tight_layout()

            # Mostra il grafico
            plt.show()

    

    def test_exact(self, solver, x_c, y_c, q, k, Lx, Ly, Nx, Ny):
        """
        Calcola l'errore RMSE per una distribuzione di carica nota.

        Args:
            solver: Oggetto solver.
            x_c (float): Coordinata x del centro della carica.
            y_c (float): Coordinata y del centro della carica.
            q (float): Carica.
            k (float): Costante.
            Lx (float): Lunghezza in x della griglia.
            Ly (float): Lunghezza in y della griglia.
            Nx (int): Numero di punti nella direzione x.
            Ny (int): Numero di punti nella direzione y.

        Returns:
            float: Errore RMSE.
        """

        methods = [self.solver.solve_gauss_seidel, self.solver.solve_jacobi, lambda grid: self.solver.solve_sor(grid, 1.8), lambda grid: self.matrix_solver.solve_poisson_equation_2d(10,5)]

        x = np.linspace(0, Lx, Nx)
        y = np.linspace(0, Ly, Ny)
        X, Y = np.meshgrid(x, y)
        squared_error_sum = 0.0
        solver.set_rho_function(self.delta_charge)

        rmse = []

        for j, method in enumerate(methods):
                grid = method(np.zeros((Nx, Ny)))

                for i in range(self.Nx):
                    for j in range(self.Ny):
                        V_known = self.known_potential(X[i, j], Y[i, j])
                        
                        V_num = grid[i,j]
                        

                        squared_error_sum += (V_known - V_num)**2

                mean_squared_error = squared_error_sum / (self.Nx * self.Ny)
                rmse[j] = np.sqrt(mean_squared_error)
        
        return rmse



    def convergence_test(self, grid_sizes):

        """
        Esegue un test di convergenza variando la dimensione della griglia.

        Args:
            grid_sizes (list): Lista delle dimensioni della griglia.

        Returns:
            list: Lista degli errori RMSE.
        """
       
        
        methods = [self.solver.solve_gauss_seidel, self.solver.solve_jacobi, lambda grid: self.solver.solve_sor(grid, 1.8), lambda grid: self.matrix_solver.solve_poisson_equation_2d(10,5)]
        errors = np.empty((len(methods), len(grid_sizes)))
        l=0
        for grid_size in grid_sizes:

            grid = np.zeros((grid_size, grid_size))

            nx = grid_size
            ny = grid_size

            solvers = PoissonDiscretization2D(nx, ny, self.Lx, self.Ly, self.epsilon_0, 100, 1e-10)
            sum = 0
            for j, method in enumerate(methods):
                for i in range(100):
                    self.x_c = np.random.uniform(0, nx)
                    self.y_c = np.random.uniform(0, ny)

                    self.q = np.random.uniform(0, 100)
                    solvers.set_rho_function(self.random_charge_distribution)

                    
                    # Risolvi il problema per la griglia corrente
                    V_num = method(grid)

                    # Calcola il potenziale noto sulla stessa griglia
                    x = np.linspace(0, self.Lx, grid_size)
                    y = np.linspace(0, self.Ly, grid_size)
                    X, Y = np.meshgrid(x, y)
                    V_known = self.known_potential_ring_charge(X, Y)
                    # Calcola l'errore RMSE
                    rmse = np.sqrt(np.mean((V_known - V_num)**2))
                    sum += rmse
                errors[j][l]= sum/100
                l+=1
            
           # Lista di colori predefiniti
            colors = ['red', 'green', 'blue', 'purple']

            # Creazione del grafico
            for j in range(len(methods)):
                # Verifica che ci siano dati nella lista errors[j] prima di plottare
                
                plt.plot(grid_sizes, errors[j], label=f'Metodo {j}', color=colors[j])
               

            # Aggiungi la legenda
            plt.legend()

            # Visualizza il grafico
            plt.show()


            

            return errors
            



     


    def test_complexity(self, grid_sizes):
        """
        Esegue un test di complessità variando la dimensione della griglia.

        Args:
            grid_sizes (list): Lista delle dimensioni della griglia.

        Returns:
            dict: Dizionario dei tempi di esecuzione per i vari metodi.
        """
        
        methods = [ self.solver.solve_gauss_seidel, self.solver.solve_jacobi, lambda grid: self.solver.solve_sor(grid, 1.8), lambda grid: self.matrix_solver.solve_poisson_equation_2d(10,5)]
        execution_times = np.empty((len(methods), len(grid_sizes)))
        for j, method in enumerate(methods):

            for index, grid_size in enumerate(grid_sizes):
                sum_execution_time = 0
                

                solvers = PoissonDiscretization2D(grid_size, grid_size,self.Lx,self.Ly,self.epsilon_0,self.num_iterations,self.tolerance)
                matrix_solvers = MultigridSolver2(grid_size, grid_size,self.Lx,self.Ly, 5,self.epsilon_0)

                for i in range(100):

                    nx = grid_size
                    ny = grid_size
                    self.x_c = np.random.uniform(0, nx)
                    self.y_c = np.random.uniform(0, ny)

                    self.q = np.random.uniform(0, 100)
                    grid = np.zeros((grid_size, grid_size))
                    solvers.set_rho_function(self.random_charge_distribution)
                    matrix_solvers.set_rho_function(self.random_charge_distribution)
                    

                    start_time = time.time()
                    grids = method(grid)
                    end_time = time.time()
                    execution_time = end_time - start_time

                    sum_execution_time += execution_time

                average_execution_time = sum_execution_time / 100
                execution_times[j][index]= average_execution_time

                print(f"Metodo: {methods}, Dimensione griglia {grid_size}x{grid_size}: Tempo di esecuzione medio = {average_execution_time} secondi")

        
        # Creazione del grafico
        for j in range(len(methods)):
            plt.plot(grid_sizes, execution_times[j], label=f'Metodo {j}')

        plt.xlabel('Dimensione Griglia')
        plt.ylabel('Tempo di Esecuzione Medio (secondi)')
        plt.legend()
        plt.show()

        return execution_times







    
    #def run_all_tests(self):




    

    
        
   

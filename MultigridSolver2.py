import numpy as np
import pandas as pd


class MultigridSolver2:
    def __init__(self, nx, ny, lx, ly, num_levels, epsilon_0):
        """
        Inizializza il solver multigrid con spazio specificato.

        Args:
            nx (int): Numero di punti sulla griglia lungo la direzione x.
            ny (int): Numero di punti sulla griglia lungo la direzione y.
            lx (float): Lunghezza dello spazio lungo la direzione x.
            ly (float): Lunghezza dello spazio lungo la direzione y.
            num_levels (int): Numero di livelli nella gerarchia multigrid.
        """
        self.nx = nx
        self.ny = ny
        self.lx = lx
        self.ly = ly
        self.num_levels = num_levels
        self.epsilon_0 = epsilon_0

        # Calcola il passo della griglia in ciascuna direzione
        self.hx = lx / (nx - 1)
        self.hy = ly / (ny - 1)

        # Inizializza la griglia più fine (livello più alto) con una griglia vuota
        self.grids = [self.create_grid(nx, ny) for _ in range(num_levels)]

        self.rho = np.zeros((nx, ny))

       

        self.dirichlet_boundaries = {'bottom': 0.0, 'top': 0.0, 'left': 0.0, 'right': 0.0}

    def normalize_initial_values(grid):
        max_val = np.max(grid)
        min_val = np.min(grid)
        scale_factor = 1 / (max_val - min_val)
        normalized_grid = (grid - min_val) * scale_factor
        return normalized_grid


    def set_rho_matrix(self, rho_matrix):
        if rho_matrix.shape == (self.nx, self.ny):
            self.rho = rho_matrix
        else:
            raise ValueError("Shape of rho_matrix must be (nx, ny)")

   

    def set_rho_function(self, rho_function):
        self.rho = np.zeros((self.nx, self.ny))
        for i in range(self.nx):
            for j in range(self.ny):
                x, y = i * self.hx, j * self.hy
                self.rho[i, j] = rho_function(x, y)
        max_rho = np.max(np.abs(self.rho))
        if max_rho > 0:
            self.rho /= max_rho

    def create_grid(self, nx, ny):
        """
        Crea una griglia vuota.

        Args:
            nx (int): Numero di punti sulla griglia lungo la direzione x.
            ny (int): Numero di punti sulla griglia lungo la direzione y.

        Returns:
            list: Lista di liste che rappresenta la griglia.
        """
        return np.zeros((nx, ny), dtype=np.float64)

    def relax(self, grid, rho, num_iterations):
        """
        Applica il metodo di Gauss-Seidel per rilassare la griglia.

        Args:
            grid (list): La griglia su cui applicare il rilassamento.
            rho (list): La griglia delle densità di carica o sorgenti.
            num_iterations (int): Numero di iterazioni del metodo di Gauss-Seidel.
        """
        for _ in range(num_iterations):
            residual = self.calculate_residual(grid, rho)
            for i in range(1, self.nx - 1):
                for j in range(1, self.ny - 1):
                    grid[i, j] += (residual[i, j] - grid[i, j] * 4.0) / 4.0

            max_phi = np.max(np.abs(grid))
            if max_phi > 0:
                grid /= max_phi 
    

            
    
    def set_zero_dirichlet_boundary(self):
        # Imposta tutti i valori delle condizioni a contorno a zero
        self.dirichlet_boundaries = {'bottom': 0.0, 'top': 0.0, 'left': 0.0, 'right': 0.0}

    
    def set_dirichlet_boundary(self, boundary=None, V=None):
        if boundary is None:
            # Se il tipo di boundary non è specificato, imposta tutti i valori a zero
            self.set_zero_dirichlet_boundary()
        elif boundary in ['bottom', 'top', 'left', 'right']:
            self.dirichlet_boundaries[boundary] = V if V is not None else 0.0
        else:
            raise ValueError("Invalid boundary. Valid options are 'bottom', 'top', 'left', 'right'.")


    def apply_dirichlet_boundary_conditions(self, grid, dirichlet_values):
        for boundary, value in dirichlet_values.items():
            if value is not None:
                if boundary == 'bottom':
                    grid[0, :] = value
                elif boundary == 'top':
                    grid[-1, :] = value
                elif boundary == 'left':
                    grid[:, 0] = value
                elif boundary == 'right':
                    grid[:, -1] = value
                
            


    def calculate_residual(self, grid, rho):
        """
        Calcola il residuo sulla griglia data.

        Args:
            grid (list): La griglia su cui calcolare il residuo.
            rho (list): La griglia delle densità di carica o sorgenti.

        Returns:
            list: Il residuo calcolato sulla griglia.
        """
        residual = np.zeros((self.nx, self.ny))
        for i in range(1, self.nx - 1):
            for j in range(1, self.ny - 1):
                x, y = i * self.hx, j * self.hy
                residual[i, j] = rho[i, j] - (grid[i + 1, j] + grid[i - 1, j] + grid[i, j + 1] + grid[i, j - 1] - 4.0 * grid[i, j]) / (self.hx**2 * self.hy**2)
        return residual

    def restrict(self, fine_grid):
        """
        Restringe il campo scalare φ da una griglia più fine a una più grossolana.

        Args:
            fine_grid (list): La griglia più fine da cui restringere il campo scalare.

        Returns:
            list: Il campo scalare φ sulla griglia più grossolana.
        """
        nx_fine, ny_fine = fine_grid.shape
        nx_coarse, ny_coarse = nx_fine // 2, ny_fine // 2

        print(f"Fine grid shape: {fine_grid.shape}")
        print(f"Coarse grid shape: {(nx_coarse, ny_coarse)}")


        # Inizializza la griglia più grossolana con valori iniziali arbitrari (0, in questo caso)
        coarse_grid = np.zeros((nx_coarse, ny_coarse))

        for i in range(0, nx_fine, 2):
                for j in range(0, ny_fine, 2):
                    # Utilizza il metodo di restrizione (Full Weighting)
                    coarse_grid[i // 2, j // 2] = 0.25 * (
                        fine_grid[i, j] + fine_grid[i + 1, j] +
                        fine_grid[i, j + 1] + fine_grid[i + 1, j + 1]
                    )
        print("Coarse grid shape after restriction:", coarse_grid.shape)



        return coarse_grid
    
    def interpolate(self, coarse_grid):
        """
        Interpola il campo scalare φ da una griglia più grossolana a una più fine.

        Args:
            coarse_grid (list): La griglia più grossolana da cui interpolare il campo scalare.

        Returns:
            list: Il campo scalare φ sulla griglia più fine.
        """
        nx_coarse, ny_coarse = coarse_grid.shape
        nx_fine, ny_fine = 2 * nx_coarse, 2 * ny_coarse

        # Inizializza la griglia più fine con valori iniziali arbitrari (0, in questo caso)
        fine_grid = np.zeros((nx_fine, ny_fine))
        

        for i in range(nx_coarse):
            for j in range(ny_coarse):

                
                # Utilizza il metodo di interpolazione bilineare
                #fine_grid[2 * i + 1, 2 * j + 1] = coarse_grid[i, j]
               
                if j < ny_coarse - 1:
                    fine_grid[2 * i, 2 * j + 1] = 0.5 * (coarse_grid[i, j] + coarse_grid[i, j + 1])
                if i < nx_coarse - 1:
                    fine_grid[2 * i + 1, 2 * j] = 0.5 * (coarse_grid[i, j] + coarse_grid[i + 1, j])
                if i < nx_coarse - 1 and j < ny_coarse - 1:
                    fine_grid[2 * i + 1, 2 * j + 1] = 0.25 * (
                        coarse_grid[i, j] + coarse_grid[i + 1, j] +
                        coarse_grid[i, j + 1] + coarse_grid[i + 1, j + 1]
                    )
                if j == ny_coarse - 1:
                    fine_grid[2 * i, 2 * j + 1] = coarse_grid[i, j]

                if i == nx_coarse - 1:
                    fine_grid[2 * i + 1, 2 * j] = coarse_grid[i, j]

       

        return fine_grid

    def solve_poisson_equation_2d(self, v_cycle_iterations=10, max_iterations=10, tolerance=1e-100, dirichlet_values=None):
            """
            Risolve l'equazione di Poisson 2D utilizzando il metodo del ciclo V.

            Args:
                rho (function): Funzione che restituisce la densità di carica o la sorgente in ogni punto (x, y).
                v_cycle_iterations (int): Numero di iterazioni del ciclo V.
                max_iterations (int): Numero massimo di iterazioni per il rilassamento di Jacobi sul livello più fine.
                tolerance (float): Tolleranza per il criterio di arresto.
                dirichlet_values (dict): Dizionario contenente i valori di condizioni di contorno di Dirichlet.

            Returns:
                np.ndarray: Il campo scalare φ calcolato sulla griglia più fine.
            """
            nx, ny = self.nx, self.ny
            hx, hy = self.hx, self.hy

             
           
            # Inizializza la griglia più fine (livello più alto) con valori iniziali arbitrari (0, in questo caso)
            phi = self.grids[-1]

            # Se il dizionario delle condizioni a contorno è fornito, utilizzalo
            self.set_dirichlet_boundary(dirichlet_values)

            # Calcola il residuo iniziale (usiamo un criterio di arresto basato sul residuo)
            residual = self.calculate_residual(phi, self.rho)

            # Esegui il ciclo V per un numero di iterazioni specificato
            for v_iteration in range(v_cycle_iterations):
                print(f"Running V-cycle iteration {v_iteration + 1}/{v_cycle_iterations}...")
                self.v_cycle(0, max_iterations, tolerance, self.rho, dirichlet_values)

            max_phi = np.max(np.abs(phi))
            if max_phi > 0:
                phi /= max_phi

            return phi

    def v_cycle(self, level, max_iterations, tolerance, rho, dirichlet_values):
        print(f"level: {level} ")
        
        # Caso base: se siamo al livello più fine, esegui il rilassamento di Jacobi
        if level == self.num_levels - 1:
            return self.solve_jacobi(self.grids[level], max_iterations, tolerance, rho, dirichlet_values)
            

        # Altrimenti, esegui il ciclo V sul livello corrente
        self.solve_jacobi(self.grids[level], max_iterations, tolerance, rho, dirichlet_values)

        # Calcola il residuo sul livello corrente
        residual = self.calculate_residual(self.grids[level], rho)

        # Interpola il residuo sul livello successivo
        residual_coarse = self.restrict(residual)

        # Richiama ricorsivamente il ciclo V sul livello successivo
        self.v_cycle(level + 1, max_iterations, tolerance, rho, dirichlet_values)

        # Interpola il campo soluzione dal livello successivo al livello corrente
        phi = self.interpolate(residual_coarse)

        # Correggi il campo soluzione sul livello corrente
        self.grids[level] += phi

        # Esegui un altro rilassamento di Jacobi sul livello corrente
        self.solve_jacobi(self.grids[level], max_iterations, tolerance, rho, dirichlet_values)

        return True


        

    def solve_jacobi(self, grid, max_iterations, tolerance, rho, dirichlet_values):
            """
            Risolve l'equazione di Poisson 2D utilizzando il metodo di rilassamento di Jacobi.

            Args:
                grid (np.ndarray): Il campo scalare φ su cui eseguire il rilassamento.
                max_iterations (int): Numero massimo di iterazioni.
                tolerance (float): Tolleranza per il criterio di arresto basato sulla differenza tra iterazioni consecutive.
                rho (np.ndarray or function): La densità di carica o la sorgente in ogni punto (x, y).
                dirichlet_values (dict): Dizionario contenente i valori di condizioni di contorno di Dirichlet.

            Returns:
                np.ndarray: Il campo scalare φ dopo il rilassamento di Jacobi.
            """
            for iteration in range(max_iterations):
                prev_grid = grid.copy()

                # Applica le condizioni di contorno di Dirichlet
                self.set_dirichlet_boundary()
                

                # Calcola il nuovo valore di φ tramite il rilassamento di Jacobi
                for i in range(1, self.nx-1):
                    for j in range(1, self.ny-1):
                        if dirichlet_values is not None and (i, j) in dirichlet_values:
                            continue

                        if i + 1 < self.nx and i - 1 >= 0:
                            phi_xx = (grid[i + 1, j] - 2 * grid[i, j] + grid[i - 1, j]) / self.hx ** 2
                        else:
                            phi_xx = 0.0

                        if j + 1 < self.ny and j - 1 >= 0:
                            phi_yy = (grid[i, j + 1] - 2 * grid[i, j] + grid[i, j - 1]) / self.hy ** 2
                        else:
                            phi_yy = 0.0

                        
                        grid[i, j] = (phi_xx + phi_yy + rho[i, j] / self.epsilon_0) / 2

                max_phi = np.max(np.abs(grid))

                if max_phi > 0:
                    grid /= max_phi 

                # Calcola la differenza tra la soluzione corrente e quella precedente
                diff = np.abs(grid - prev_grid)

                # Verifica la convergenza
                if np.max(diff) < tolerance:
                    print(f"Jacobi converged after {iteration + 1} iterations.")
                    break

            return grid


        
          

        



   

   


 
import numpy as np


class MultigridSolver2:
    def __init__(self, nx, ny, lx, ly, num_levels):
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

        # Calcola il passo della griglia in ciascuna direzione
        self.hx = lx / (nx - 1)
        self.hy = ly / (ny - 1)

        # Inizializza la griglia più fine (livello più alto) con una griglia vuota
        self.grids = [self.create_grid(nx, ny) for _ in range(num_levels)]

    def create_grid(self, nx, ny):
        """
        Crea una griglia vuota.

        Args:
            nx (int): Numero di punti sulla griglia lungo la direzione x.
            ny (int): Numero di punti sulla griglia lungo la direzione y.

        Returns:
            list: Lista di liste che rappresenta la griglia.
        """
        return np.zeros((nx, ny))

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
    
    
    def set_dirichlet_boundary_conditions(self, grid):
        # Imposta i valori del potenziale φ a zero sui bordi della griglia
        grid[0, :] = 0.0  # Prima riga
        grid[-1, :] = 0.0  # Ultima riga
        grid[:, 0] = 0.0  # Prima colonna
        grid[:, -1] = 0.0  # Ultima colonna

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
                residual[i, j] = rho(x, y) - (grid[i + 1, j] + grid[i - 1, j] + grid[i, j + 1] + grid[i, j - 1] - 4.0 * grid[i, j]) / (self.hx**2 * self.hy**2)
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

        for i in range(nx_fine):
            fine_grid[i, 0] = fine_grid[i, 1]
            fine_grid[i, -1] = fine_grid[i, -2]
    
        for j in range(ny_fine):
            fine_grid[0, j] = fine_grid[1, j]
            fine_grid[-1, j] = fine_grid[-2, j]

        return fine_grid

        
          

        



    

    def v_cycle(self, level, max_iterations, tolerance, rho):
        if level == self.num_levels - 1:
            # Livello più fine (livello più alto), usa il rilassamento Gauss-Seidel
            phi = self.grids[level]
            for iteration in range(max_iterations):
                self.relax(phi, rho, num_iterations=1)
                residual = self.calculate_residual(phi, rho)
                if np.linalg.norm(residual) < tolerance:
                    print(f"Converged in {iteration} iterations on level {level}.")
                    break
        else:
            # Altrimenti, esegui un ciclo ricorsivo su una griglia più grossolana
            coarse_grid = self.restrict(self.grids[level + 1])
            

            self.v_cycle(level + 1, max_iterations, tolerance, rho)
            phi = self.grids[level]
            for iteration in range(max_iterations):
                self.relax(phi, rho, num_iterations=1)
                residual = self.calculate_residual(phi, rho)
                if np.linalg.norm(residual) < tolerance:
                    print(f"Converged in {iteration} iterations on level {level}.")
                    break

            # Interpolazione e correzione
            fine_grid = self.grids[level]
            
            interpolated_coarse_grid = self.interpolate(coarse_grid)
            fine_grid += interpolated_coarse_grid


    



    def solve_poisson_equation_2d(self, rho, v_cycle_iterations=4, max_iterations=100, tolerance=1e-6):
        """
        Risolve l'equazione di Poisson 2D utilizzando il metodo del ciclo V.

        Args:
            rho (function): Funzione che restituisce la densità di carica o la sorgente in ogni punto (x, y).
            v_cycle_iterations (int): Numero di iterazioni del ciclo V.
            max_iterations (int): Numero massimo di iterazioni per il rilassamento di Jacobi sul livello più fine.
            tolerance (float): Tolleranza per il criterio di arresto.

        Returns:
            np.ndarray: Il campo scalare φ calcolato sulla griglia più fine.
        """
        nx, ny = self.nx, self.ny
        hx, hy = self.hx, self.hy

        # Inizializza la griglia più fine (livello più alto) con valori iniziali arbitrari (0, in questo caso)
        phi = self.grids[-1]

        self.set_dirichlet_boundary_conditions(phi)

        # Calcola il residuo iniziale (usiamo un criterio di arresto basato sul residuo)
        residual = self.calculate_residual(phi, rho)

        
        # Esegui il ciclo V per un numero di iterazioni specificato
        for v_iteration in range(v_cycle_iterations):
            
            print(f"Running V-cycle iteration {v_iteration + 1}/{v_cycle_iterations}...")
            self.v_cycle(2, max_iterations, tolerance, rho)
            

            

        return phi







   

   


 
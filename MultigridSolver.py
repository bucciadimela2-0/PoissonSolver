import numpy as np
from scipy.sparse import diags


class MultigridSolver:
    def __init__(self, Nx, Ny, Lx, Ly, epsilon_0, num_iterations, tolerance):
        # Inizializza i parametri per la discretizzazione dello spazio
        self.Nx = Nx
        self.Ny = Ny
        self.Lx = Lx
        self.Ly = Ly
        self.epsilon_0 = epsilon_0
        self.dx = Lx / (Nx - 1)
        self.dy = Ly / (Ny - 1)
        self.phi = np.zeros((Nx, Ny))
        self.rho = np.zeros((Nx, Ny))
        self.dirichlet_boundaries = {}

        # Inizializza i parametri per il metodo multigrid
        self.num_iterations = num_iterations
        self.tolerance = tolerance

    def set_rho(self, rho):
        # Imposta la densità di carica all'interno del dominio
        if rho.shape == (self.Nx, self.Ny):
            self.rho = rho
        else:
            raise ValueError("Shape of rho must be (Nx, Ny)")

    def set_dirichlet_boundary(self, boundary, V):
        # Imposta le condizioni al contorno di Dirichlet su uno dei lati del dominio
        if boundary in ['bottom', 'top', 'left', 'right']:
            if boundary == 'bottom':
                self.phi[:, 0] = V
            elif boundary == 'top':
                self.phi[:, -1] = V
            elif boundary == 'left':
                self.phi[0, :] = V
            elif boundary == 'right':
                self.phi[-1, :] = V
        else:
            raise ValueError("Invalid boundary. Valid options are 'bottom', 'top', 'left', 'right'.")

    def apply_dirichlet_boundary_conditions(self):
        # Applica le condizioni al contorno di Dirichlet sulla griglia phi
        for boundary, V in self.dirichlet_boundaries.items():
            if boundary == 'bottom':
                self.phi[:, 0] = V
            elif boundary == 'top':
                self.phi[:, -1] = V
            elif boundary == 'left':
                self.phi[0, :] = V
            elif boundary == 'right':
                self.phi[-1, :] = V

    def apply_neumann_boundary_conditions(self):
        # Applica le condizioni al contorno di Neumann sulla griglia phi
        # Questo esempio imposta i gradienti della soluzione ai lati del dominio a zero (derivata zero)
        self.phi[0, :] = self.phi[1, :]
        self.phi[-1, :] = self.phi[-2, :]
        self.phi[:, 0] = self.phi[:, 1]
        self.phi[:, -1] = self.phi[:, -2]

    def apply_dirichlet_boundary_conditions_to_grid(self, grid):
        # Applica le condizioni al contorno di Dirichlet sulla griglia fornita come parametro in input
        for boundary, V in self.dirichlet_boundaries.items():
            if boundary == 'bottom':
                grid[:, 0] = V
            elif boundary == 'top':
                grid[:, -1] = V
            elif boundary == 'left':
                grid[0, :] = V
            elif boundary == 'right':
                grid[-1, :] = V

    def compute_residual(self, grid):
        # Calcola il residuo del sistema Poisson discretizzato
        # Lato sinistro: laplaciano del potenziale elettrostatico (operatore discretizzato)
        # Lato destro: densità di carica moltiplicata per la costante dielettrica nel vuoto

        # Calcola il laplaciano del potenziale elettrostatico
        laplacian = np.zeros_like(grid)
        Nx, Ny = grid.shape
        for i in range(1, Nx - 1):
            for j in range(1, Ny - 1):
                laplacian[i, j] = (
                    grid[i + 1, j] + grid[i - 1, j] +
                    grid[i, j + 1] + grid[i, j - 1] -
                    4 * grid[i, j]
                ) / (self.dx ** 2)

        # Calcola il lato destro dell'equazione di Poisson discretizzata
        rhs = self.epsilon_0 * self.rho

        # Calcola il residuo come differenza tra il lato sinistro e il lato destro
        residual = laplacian - rhs

        return residual

    def restrict(self, fine_grid):
        Nx_coarse = (self.Nx - 1) // 2 + 1
        Ny_coarse = (self.Ny - 1) // 2 + 1
        # Calcola la dimensione della griglia più grossolana
        coarse_shape = ((self.Nx - 1) // 2 + 1, (self.Ny - 1) // 2 + 1)

        # Inizializza la griglia più grossolana
        coarse_grid = np.zeros(coarse_shape)
        coarse_rho = np.zeros((Nx_coarse, Ny_coarse))

        # Riduci il residuo dalla griglia più fine alla griglia più grossolana
        for i in range(coarse_shape[0]):
            for j in range(coarse_shape[1]):
                # Calcola gli indici corrispondenti nella griglia più fine
                i_fine = 2 * i
                j_fine = 2 * j

                # Assicurati che gli indici siano all'interno dei limiti della griglia più fine
                if i_fine < fine_grid.shape[0] - 1 and j_fine < fine_grid.shape[1] - 1:
                    # Calcola il valore ridotto utilizzando solo i punti interni della griglia più fine
                    coarse_grid[i, j] = (
                        fine_grid[i_fine, j_fine] +
                        fine_grid[i_fine + 1, j_fine] +
                        fine_grid[i_fine, j_fine + 1] +
                        fine_grid[i_fine + 1, j_fine + 1]
                    ) * 0.25

                    coarse_rho[i, j] = (
                    self.rho[i_fine, j_fine] +
                    self.rho[i_fine + 1, j_fine] +
                    self.rho[i_fine, j_fine + 1] +
                    self.rho[i_fine + 1, j_fine + 1]
                     ) * 0.25
                else:
                    # Se gli indici sono fuori dai limiti, gestisci i punti di bordo
                    if i_fine >= fine_grid.shape[0] - 1:
                        i_fine = fine_grid.shape[0] - 2
                    if j_fine >= fine_grid.shape[1] - 1:
                        j_fine = fine_grid.shape[1] - 2

                    coarse_grid[i, j] = fine_grid[i_fine, j_fine]
                    coarse_rho[i, j] = self.rho[i_fine, j_fine]

        self.rho = coarse_rho
        # Applica le condizioni al contorno di Dirichlet sulla griglia più grossolana
        self.apply_dirichlet_boundary_conditions_to_grid(coarse_grid)

        return coarse_grid

    def interpolate(self, coarse_grid):
        # Calcola la dimensione della griglia più fine
        fine_shape = (coarse_grid.shape[0] * 2 - 1, coarse_grid.shape[1] * 2 - 1)

        # Inizializza la griglia più fine con valori NaN
        fine_grid = np.full(fine_shape, np.nan)

        # Copia la griglia più grossolana nella posizione corrispondente della griglia più fine
        fine_grid[::2, ::2] = coarse_grid

        # Interpola la soluzione dalla griglia più grossolana alla griglia più fine
        for i in range(1, fine_shape[0] - 1, 2):
            for j in range(1, fine_shape[1] - 1, 2):
                if not np.isnan(fine_grid[i, j]):
                    # Interpola solo i punti interni della griglia più grossolana
                    fine_grid[i, j + 1] = 0.5 * (fine_grid[i, j] + fine_grid[i, j + 2])
                    fine_grid[i + 1, j] = 0.5 * (fine_grid[i, j] + fine_grid[i + 2, j])
                    fine_grid[i + 1, j + 1] = 0.25 * (fine_grid[i, j] + fine_grid[i + 2, j] + fine_grid[i, j + 2] + fine_grid[i + 2, j + 2])

        # Applica le condizioni al contorno di Dirichlet sulla griglia più fine
        self.apply_dirichlet_boundary_conditions_to_grid(fine_grid)

        # Riduci la dimensione della griglia più fine se necessario
        if fine_shape[0] > self.Nx or fine_shape[1] > self.Ny:
            fine_grid = fine_grid[:self.Nx, :self.Ny]

        return fine_grid




    def compute_A(self, grid):
        # Costruisci la matrice tridiagonale associata all'operatore Laplaciano
        Nx, Ny = grid.shape
        main_diag = -4 * np.ones(Nx * Ny)
        off_diag = np.ones(Nx * Ny)
        off_diag[Ny::Ny] = 0  # Imposta gli elementi off-diagonali sulla riga Ny a zero
        off_diag[Ny - 1::Ny] = 0  # Imposta gli elementi off-diagonali sulla riga Ny - 1 a zero
        A = diags([off_diag, main_diag, off_diag], [-1, 0, 1], shape=(Nx * Ny, Nx * Ny), format='csr')

        # Calcola il prodotto matrice-vettore A * phi
        flattened_phi = grid.ravel()
        Ax = A.dot(flattened_phi)

        # Reshape il risultato come griglia
        Ax = Ax.reshape((Nx, Ny))

        return Ax


    def is_converged(self, residual):
        # Calcola la norma infinita del residuo
        residual_norm = np.max(np.abs(residual))

        # Verifica se la norma del residuo è inferiore alla tolleranza
        return residual_norm < self.tolerance

    def discretize(self, grid, omega):
        # Copia la griglia corrente in una nuova variabile (questo dipende dal metodo SOR specifico)
        new_grid = grid.copy()

        # Parametro di tolleranza per il criterio di convergenza (scegli un valore appropriato)
        tolerance = 1e-6

        # Numero massimo di iterazioni (scegli un valore appropriato)
        max_iterations = 1000

        # Dimensione della griglia (assumendo che sia una griglia regolare)
        nx, ny = grid.shape

        for iteration in range(max_iterations):
            # Inizializza la norma del residuo a zero
            residual_norm = 0.0

            # Itera attraverso i punti interni della griglia
            for i in range(1, nx - 1):
                for j in range(1, ny - 1):
                    # Calcola il nuovo valore del punto sulla griglia utilizzando il metodo SOR
                    new_value = (1 - omega) * grid[i, j] + omega * (
                        (grid[i - 1, j] + grid[i + 1, j] + grid[i, j - 1] + grid[i, j + 1]) / 4
                    )

                    # Calcola il residuo per il punto corrente
                    residual = abs(new_value - grid[i, j])

                    # Aggiorna il valore sulla griglia
                    new_grid[i, j] = new_value

                    # Aggiorna la norma del residuo
                    residual_norm += residual

            # Calcola la norma del residuo normalizzata
            residual_norm /= (nx - 2) * (ny - 2)

            # Verifica il criterio di convergenza
            if residual_norm < tolerance:
                break

            # Copia i nuovi valori nella griglia corrente per la prossima iterazione
            grid[:, :] = new_grid

        return grid

    def full_multigrid(self, grid, num_levels, omega):
        if num_levels == 1:
            # Raggiunto il livello più grosso, risolvi con il metodo SOR
            return self.discretize(grid, omega)

        # Pre-smoothing con il metodo SOR sulla griglia più fine
        grid = self.discretize(grid, omega)

        # Calcola il residuo r = b - Ax
        residual = self.compute_residual(grid)

        # Riduci il residuo sulla griglia più grossolana utilizzando l'operazione di restrizione
        coarse_residual = self.restrict(residual)

        # Inizializza la soluzione approssimata su una griglia più grossolana
        coarse_phi = np.zeros_like(coarse_residual)

        # Ricorsivamente risolvi il problema sulla griglia più grossolana
        coarse_phi = self.full_multigrid(coarse_phi, num_levels - 1, omega)

        # Interpola la soluzione approssimata sull'errore corrente
        interpolated_error = self.interpolate(coarse_phi)

        # Correggi la soluzione sulla griglia più fine
        grid[1:-1, 1:-1] += interpolated_error[1:-1, 1:-1]

        # Post-smoothing con il metodo SOR sulla griglia più fine
        grid = self.discretize(grid, omega)

        return grid

    def solve(self, num_levels, omega):
        # Esegui il metodo Full Multigrid
        self.phi = self.full_multigrid(self.phi, num_levels, omega)

        for iteration in range(self.num_iterations):
            # Applica le condizioni al contorno
            self.apply_dirichlet_boundary_conditions()
            self.apply_neumann_boundary_conditions()

            # Calcola il residuo
            residual = self.compute_residual(self.phi)

            # Verifica la convergenza
            if self.is_converged(residual):
                print(f"Convergenza raggiunta dopo {iteration + 1} iterazioni.")
                break

        return self.phi

   


    



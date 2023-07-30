
import numpy as np


class PoissonDiscretization2D:
    def __init__(self, Nx, Ny, Lx, Ly, epsilon_0,num_iterations,tolerance):
        #Dimensioni della griglia di discetizzazione, rispetto x e y
            self.Nx = Nx
            self.Ny = Ny
        #Lunghezza del dominio
            self.Lx = Lx
            self.Ly = Ly
        #Costante dielettrica nel vuoto
            self.epsilon_0 = epsilon_0
        #Intervalli di spaziatura tra i punti discreti
            self.dx = Lx / (Nx - 1)
            self.dy = Ly / (Ny - 1)
        #funzioni potenziale elettrostatico e densità di carica
            self.phi = np.zeros((Nx, Ny))
            self.rho = np.zeros((Nx, Ny))
        #dizionario che memorizza le condizioni di Dirichlet specificate per i lati del dominio
            self.dirichlet_boundaries = {}
            self.num_iterations = num_iterations
        #tolleranza utilizzata per verificare la convergenza del solver
            self.tolerance = tolerance

    #imposta la densità di carica all'interno del dominio
    def set_rho(self, rho):
        #Controlla la dimensione della matrice
        if rho.shape == (self.Nx, self.Ny):
            self.rho = rho
        else:
            raise ValueError("Shape of rho must be (Nx, Ny)")
   
   
    


    def set_dirichlet_boundary(self, boundary, V):
        if boundary in ['bottom', 'top', 'left', 'right']:
            self.dirichlet_boundaries[boundary] = V
        else:
            raise ValueError("Invalid boundary. Valid options are 'bottom', 'top', 'left', 'right'.")

    #applica le condizioni di Dirichlet ai lati del dominio specificati tramite il dizionario self.dirichlet_boundaries
    def apply_dirichlet_boundary_conditions(self):
        for boundary, V in self.dirichlet_boundaries.items():
            if boundary == 'bottom':
                self.phi[:, 0] = V
            elif boundary == 'top':
                self.phi[:, -1] = V
            elif boundary == 'left':
                self.phi[0, :] = V
            elif boundary == 'right':
                self.phi[-1, :] = V
    #applica le condizioni di Neumann ai lati del dominio
    def apply_neumann_boundary_conditions(self):
        self.phi[0, :] = self.phi[1, :]
        self.phi[-1, :] = self.phi[-2, :]
        self.phi[:, 0] = self.phi[:, 1]
        self.phi[:, -1] = self.phi[:, -2]


    def discretize(self):
        for i in range(self.Nx):
            for j in range(self.Ny):
                phi_xx = (self.phi[min(i+1, self.Nx-1), j] - 2 * self.phi[i, j] + self.phi[max(i-1, 0), j]) / self.dx**2
                phi_yy = (self.phi[i, min(j+1, self.Ny-1)] - 2 * self.phi[i, j] + self.phi[i, max(j-1, 0)]) / self.dy**2
                self.phi[i, j] = (phi_xx + phi_yy + self.rho[i, j] / self.epsilon_0) / 2
        return self.phi

  

    def solve_gauss_seidel(self, grid):
        self.phi = grid.copy()  # Utilizziamo la griglia fornita come punto di partenza

        for iteration in range(self.num_iterations):
            prev_phi = np.copy(self.phi)

            for i in range(1, self.Nx - 1):
                for j in range(1, self.Ny - 1):
                    phi_xx = (self.phi[i + 1, j] - 2 * self.phi[i, j] + self.phi[i - 1, j]) / self.dx**2
                    phi_yy = (self.phi[i, j + 1] - 2 * self.phi[i, j] + self.phi[i, j - 1]) / self.dy**2
                    self.phi[i, j] = (phi_xx + phi_yy + self.rho[i, j] / self.epsilon_0) / 2

            # Applica le condizioni di Dirichlet ai lati del dominio
            self.apply_dirichlet_boundary_conditions()

            # Verifica la convergenza
            if np.max(np.abs(prev_phi - self.phi)) < self.tolerance:
                print(f"Gauss-Seidel converged after {iteration + 1} iterations.")
                break

        return self.phi

    def solve_jacobi(self, grid):
        self.phi = grid.copy()  # Utilizziamo la griglia fornita come punto di partenza

        for iteration in range(self.num_iterations):
            prev_phi = np.copy(self.phi)

            for i in range(1, self.Nx - 1):
                for j in range(1, self.Ny - 1):
                    phi_xx = (prev_phi[i + 1, j] - 2 * prev_phi[i, j] + prev_phi[i - 1, j]) / self.dx**2
                    phi_yy = (prev_phi[i, j + 1] - 2 * prev_phi[i, j] + prev_phi[i, j - 1]) / self.dy**2
                    self.phi[i, j] = (phi_xx + phi_yy + self.rho[i, j] / self.epsilon_0) / 2

            # Applica le condizioni di Dirichlet ai lati del dominio
            self.apply_dirichlet_boundary_conditions()

            # Verifica la convergenza
            if np.max(np.abs(prev_phi - self.phi)) < self.tolerance:
                print(f"Jacobi converged after {iteration + 1} iterations.")
                break

        return self.phi

    def solve_sor(self, grid, omega):
            self.phi = grid.copy()  # Utilizziamo la griglia fornita come punto di partenza

            for iteration in range(self.num_iterations):
                prev_phi = np.copy(self.phi)

                for i in range(1, self.Nx - 1):
                    for j in range(1, self.Ny - 1):
                        phi_xx = (prev_phi[i + 1, j] - 2 * prev_phi[i, j] + prev_phi[i - 1, j]) / self.dx**2
                        phi_yy = (prev_phi[i, j + 1] - 2 * prev_phi[i, j] + prev_phi[i, j - 1]) / self.dy**2
                        self.phi[i, j] = (1 - omega) * prev_phi[i, j] + omega * ((phi_xx + phi_yy + self.rho[i, j] / self.epsilon_0) / 2)

                # Applica le condizioni di Dirichlet ai lati del dominio
                self.apply_dirichlet_boundary_conditions()

                # Verifica la convergenza
                if np.max(np.abs(prev_phi - self.phi)) < self.tolerance:
                    print(f"SOR (omega={omega}) converged after {iteration + 1} iterations.")
                    break

            return self.phi


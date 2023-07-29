
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
    
    #Discretizzazione dell'equazione di poisson
    def discretize(self):
        for i in range(self.num_iterations):
            prev_phi = np.copy(self.phi)

             # Iterazione dell'equazione di Poisson utilizzando il metodo delle differenze finite
            for i in range(1, self.Nx - 1):
                for j in range(1, self.Ny - 1):
                    phi_xx = (self.phi[i + 1, j] - 2 * self.phi[i, j] + self.phi[i - 1, j]) / self.dx**2
                    phi_yy = (self.phi[i, j + 1] - 2 * self.phi[i, j] + self.phi[i, j - 1]) / self.dy**2
                    self.phi[i, j] = (phi_xx + phi_yy + self.rho[i, j] / self.epsilon_0) / 2

             # Applicazione delle condizioni al contorno
            if self.dirichlet_boundaries:
                self.apply_dirichlet_boundary_conditions()
            else:
                 self.apply_neumann_boundary_conditions()

             # Verifica della convergenza
            if np.max(np.abs(self.phi - prev_phi)) < self.tolerance:
                break

        return self.phi

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
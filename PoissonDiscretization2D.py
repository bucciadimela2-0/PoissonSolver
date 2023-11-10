
import numpy as np


class PoissonDiscretization2D:
    def __init__(self, Nx, Ny, Lx, Ly, epsilon_0,num_iterations,tolerance):
            """
            Inizializza l'oggetto PoissonDiscretization2D.

            Args:
                Nx (int): Numero di punti discreti nella direzione x.
                Ny (int): Numero di punti discreti nella direzione y.
                Lx (float): Lunghezza del dominio nella direzione x.
                Ly (float): Lunghezza del dominio nella direzione y.
                epsilon_0 (float): Costante dielettrica nel vuoto.
                num_iterations (int): Numero massimo di iterazioni per i solutori iterativi.
                tolerance (float): Tolleranza per la convergenza del solver.
            """
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
            self.dirichlet_boundaries = {'bottom': 0.0, 'top': 0.0, 'left': 0.0, 'right': 0.0}
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
    
    def set_rho_function(self, rho_function):
        """
        Imposta la densità di carica in base a una funzione specificata.

        Args:
            rho_function (function): Funzione che restituisce la densità di carica in base alle coordinate (x, y).
        """
        self.rho = np.zeros((self.Nx, self.Ny))
        for i in range(self.Nx):
            for j in range(self.Ny):
                x, y = i * self.dx, j * self.dy
                self.rho[i, j] = rho_function(x, y)
        max_rho = np.max(np.abs(self.rho))
        if max_rho > 0:
            self.rho /= max_rho
   
   
    
    def set_dirichlet_boundary(self, boundary=None, V=None):
        """
        Imposta le condizioni di Dirichlet su uno dei lati del dominio.

        Args:
            boundary (str): Specifica il lato del dominio ('bottom', 'top', 'left', 'right').
            V (float): Valore del potenziale.

        Raises:
            ValueError: Se il lato specificato non è valido.
        """
        if boundary is None:
            # Se il tipo di boundary non è specificato, imposta il valore di default a zero per tutti i lati
            self.dirichlet_boundaries = {'bottom': 0.0, 'top': 0.0, 'left': 0.0, 'right': 0.0}
        elif boundary in ['bottom', 'top', 'left', 'right']:
            self.dirichlet_boundaries[boundary] = V if V is not None else 0.0
        else:
            raise ValueError("Invalid boundary. Valid options are 'bottom', 'top', 'left', 'right'.")


    def apply_dirichlet_boundary_conditions(self):
        """
        Applica le condizioni di Dirichlet al potenziale elettrico sulla griglia.
        """
        for boundary, value in self.dirichlet_boundaries.items():
            if value is not None:
                if boundary == 'bottom':
                    self.phi[:, 0] = value
                elif boundary == 'top':
                    self.phi[:, -1] = value
                elif boundary == 'left':
                    self.phi[0, :] = value
                elif boundary == 'right':
                    self.phi[-1, :] = value

    


    def discretize(self):
        for i in range(self.Nx):
            for j in range(self.Ny):
                phi_xx = (self.phi[min(i+1, self.Nx-1), j] - 2 * self.phi[i, j] + self.phi[max(i-1, 0), j]) / self.dx**2
                phi_yy = (self.phi[i, min(j+1, self.Ny-1)] - 2 * self.phi[i, j] + self.phi[i, max(j-1, 0)]) / self.dy**2
                self.phi[i, j] = (phi_xx + phi_yy + self.rho[i, j] / self.epsilon_0) / 2
         # Normalizzazione dei valori del potenziale
       
        max_phi = np.max(np.abs(self.phi))
        if max_phi > 0:
            self.phi /= max_phi

        
        return self.phi

  

    def solve_gauss_seidel(self, grid):
        """
        Risolve l'equazione di Poisson utilizzando il metodo iterativo di Gauss-Seidel.

        Args:
            grid (numpy.ndarray): Griglia iniziale del potenziale.

        Returns:
            numpy.ndarray: Griglia del potenziale dopo la soluzione.
        """
        self.phi = grid.copy()

        for iteration in range(self.num_iterations):
            prev_phi = self.phi.copy()

            for i in range(1, self.Nx - 1):
                for j in range(1, self.Ny - 1):
                    phi_xx = (self.phi[min(i+1, self.Nx-1), j] - 2 * self.phi[i, j] + self.phi[max(i-1, 0), j]) / self.dx**2
                    phi_yy = (self.phi[i, min(j+1, self.Ny-1)] - 2 * self.phi[i, j] + self.phi[i, max(j-1, 0)]) / self.dy**2

                    if i == 0 or i == self.Nx - 1 or j == 0 or j == self.Ny - 1:
                        # Se siamo su un bordo, applica la condizione di Dirichlet
                        self.phi[i, j] = self.dirichlet_boundaries.get('bottom', 0.0) if j == 0 else \
                                        self.dirichlet_boundaries.get('top', 0.0) if j == self.Ny - 1 else \
                                        self.dirichlet_boundaries.get('left', 0.0) if i == 0 else \
                                        self.dirichlet_boundaries.get('right', 0.0)
                    else:
                        self.phi[i, j] = (phi_xx + phi_yy + self.rho[i, j] / self.epsilon_0) / 2

            max_phi = np.max(np.abs(self.phi))
            if max_phi > 0:
                self.phi /= max_phi

            if np.max(np.abs(prev_phi - self.phi)) < self.tolerance:
                print(f"Gauss-Seidel converged after {iteration + 1} iterations.")
                break

        return self.phi




            

        

    def solve_jacobi(self, grid):
        """
        Risolve l'equazione di Poisson utilizzando il metodo iterativo di Jacobi.

        Args:
            grid (numpy.ndarray): Griglia iniziale del potenziale.

        Returns:
            numpy.ndarray: Griglia del potenziale dopo la soluzione.
        """
        self.phi = grid.copy()  # Utilizziamo la griglia fornita come punto di partenza

        for iteration in range(self.num_iterations):
            prev_phi = np.copy(self.phi)

            for i in range(1, self.Nx - 1):
                for j in range(1, self.Ny - 1):
                    phi_xx = (prev_phi[i + 1, j] - 2 * prev_phi[i, j] + prev_phi[i - 1, j]) / self.dx**2
                    phi_yy = (prev_phi[i, j + 1] - 2 * prev_phi[i, j] + prev_phi[i, j - 1]) / self.dy**2
                    self.phi[i, j] = (phi_xx + phi_yy + self.rho[i, j] / self.epsilon_0) / 2
            
             # Normalizzazione dei valori del potenziale
            max_phi = np.max(np.abs(self.phi))
            if max_phi > 0:
                self.phi /= max_phi

            # Applica le condizioni di Dirichlet ai lati del dominio
            self.apply_dirichlet_boundary_conditions()

            # Verifica la convergenza
            if np.max(np.abs(prev_phi - self.phi)) < self.tolerance:
                print(f"Jacobi converged after {iteration + 1} iterations.")
                break

        return self.phi

    def solve_sor(self, grid, omega):

            """
            Risolve l'equazione di Poisson utilizzando il metodo iterativo SOR (Successive Over-Relaxation).

            Args:
                grid (numpy.ndarray): Griglia iniziale del potenziale.
                omega (float): Fattore di rilassamento.

            Returns:
                numpy.ndarray: Griglia del potenziale dopo la soluzione.
            """
            self.phi = grid.copy()  # Utilizziamo la griglia fornita come punto di partenza

            for iteration in range(self.num_iterations):
                prev_phi = np.copy(self.phi)

                for i in range(1, self.Nx - 1):
                    for j in range(1, self.Ny - 1):
                        phi_xx = (prev_phi[i + 1, j] - 2 * prev_phi[i, j] + prev_phi[i - 1, j]) / self.dx**2
                        phi_yy = (prev_phi[i, j + 1] - 2 * prev_phi[i, j] + prev_phi[i, j - 1]) / self.dy**2
                        self.phi[i, j] = (1 - omega) * prev_phi[i, j] + omega * ((phi_xx + phi_yy + self.rho[i, j] / self.epsilon_0) / 2)

                 # Normalizzazione dei valori del potenziale
                max_phi = np.max(np.abs(self.phi))
                if max_phi > 0:
                    self.phi /= max_phi
                    # Applica le condizioni di Dirichlet ai lati del dominio
                    self.apply_dirichlet_boundary_conditions()

                # Verifica la convergenza
                if np.max(np.abs(prev_phi - self.phi)) < self.tolerance:
                    print(f"SOR (omega={omega}) converged after {iteration + 1} iterations.")
                    break

            return self.phi


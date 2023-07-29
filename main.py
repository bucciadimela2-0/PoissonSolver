import matplotlib.pyplot as plt
import numpy as np

from PoissonDiscretization2D import PoissonDiscretization2D

if __name__ == "__main__":
   
    Nx = 10
    Ny = 10
    Lx = 100.0
    Ly = 100.0
    epsilon_0 = 8.854187817e-12  # Costante dielettrica del vuoto


    solver = PoissonDiscretization2D(Nx, Ny, Lx, Ly, epsilon_0,100,1e-6)

    # Imposta le condizioni al contorno
    V_bottom = 0.0
    V_top = 0.0
    solver.set_dirichlet_boundary('bottom', V_bottom)
    solver.set_dirichlet_boundary('top', V_top)

    # Imposta la densità di carica
    rho = np.ones((Nx, Ny))  # Densità di carica uniforme (esempio)
    solver.set_rho(rho)

    # Risolvi l'equazione di Poisson
    phi_solution = solver.discretize()

    # Visualizza la soluzione del potenziale elettrostatico
    plt.imshow(phi_solution, cmap='jet', origin='lower', extent=[0, Lx, 0, Ly])
    plt.colorbar(label='Potenziale elettrostatico')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Discretizzazione del campo elettrostatico')
    plt.show()



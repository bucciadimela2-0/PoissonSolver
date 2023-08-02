import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from MultigridSolver2 import MultigridSolver2
from PoissonDiscretization2D import PoissonDiscretization2D

if __name__ == "__main__":
   
    Nx = 100
    Ny = 100
    Lx = 100.0
    Ly = 100.0
    epsilon_0 = 8.854187817e-12  # Costante dielettrica del vuoto


    solver = PoissonDiscretization2D(Nx, Ny, Lx, Ly, epsilon_0,100,1e-6)

    # Imposta le condizioni al contorno
    V_bottom = 100.0
    V_top = 100.0
    solver.set_dirichlet_boundary('bottom', V_bottom)
    solver.set_dirichlet_boundary('top', V_top)

    # Imposta la densità di carica
    rho = np.ones((Nx, Ny))  # Densità di carica uniforme (esempio)
    solver.set_rho(rho)

    # Risolvi l'equazione di Poisson
    phi_solution = solver.discretize()
   
    print(phi_solution)
    plt.imshow(phi_solution, cmap='jet', origin='lower', extent=[0, Lx, 0, Ly])

        # Converti l'array numpy in un DataFrame di pandas per visualizzare come tabella
    df = pd.DataFrame(phi_solution, columns=[f'y={i*Ly/(Ny-1):.2e}' for i in range(Ny)],
                    index=[f'x={i*Lx/(Nx-1):.2e}' for i in range(Nx)])
    
    with open('potential_table.txt', 'w') as file:
        file.write(df.to_string())

    # Visualizza la tabella dei valori del potenziale elettrostatico con notazione scientifica
    print(df)


    plt.colorbar(label='Potenziale elettrostatico')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Griglia discretizzata del potenziale elettrostatico')
    plt.grid(color='black', linestyle='--', linewidth=0.5)
    plt.show()


    grid = solver.solve_gauss_seidel(phi_solution)
    plt.imshow(grid, cmap='jet', origin='lower', extent=[0, Lx, 0, Ly])

        # Converti l'array numpy in un DataFrame di pandas per visualizzare come tabella
    df = pd.DataFrame(grid, columns=[f'y={i*Ly/(Ny-1):.2e}' for i in range(Ny)],
                    index=[f'x={i*Lx/(Nx-1):.2e}' for i in range(Nx)])
    
    with open('Gauss_Siedel_table.txt', 'w') as file:
        file.write(df.to_string())

    # Visualizza la tabella dei valori del potenziale elettrostatico con notazione scientifica
    print(df)


    plt.colorbar(label='Gauss Siedel')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Griglia discretizzata del potenziale elettrostatico')
    plt.grid(color='black', linestyle='--', linewidth=0.5)
    plt.show()
  

    grid = solver.solve_jacobi(phi_solution)

    plt.imshow(grid, cmap='jet', origin='lower', extent=[0, Lx, 0, Ly])

        # Converti l'array numpy in un DataFrame di pandas per visualizzare come tabella
    df = pd.DataFrame(grid, columns=[f'y={i*Ly/(Ny-1):.2e}' for i in range(Ny)],
                    index=[f'x={i*Lx/(Nx-1):.2e}' for i in range(Nx)])
    
    with open('Jacobi_table.txt', 'w') as file:
        file.write(df.to_string())

    # Visualizza la tabella dei valori del potenziale elettrostatico con notazione scientifica
    print(df)


    plt.colorbar(label='Gauss Siedel')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Jacobi')
    plt.grid(color='black', linestyle='--', linewidth=0.5)
    plt.show()


    grid = solver.solve_sor(phi_solution,2)

    plt.imshow(grid, cmap='jet', origin='lower', extent=[0, Lx, 0, Ly])

        # Converti l'array numpy in un DataFrame di pandas per visualizzare come tabella
    df = pd.DataFrame(grid, columns=[f'y={i*Ly/(Ny-1):.2e}' for i in range(Ny)],
                    index=[f'x={i*Lx/(Nx-1):.2e}' for i in range(Nx)])
    
    with open('Sor_table.txt', 'w') as file:
        file.write(df.to_string())

    # Visualizza la tabella dei valori del potenziale elettrostatico con notazione scientifica
    print(df)


    plt.colorbar(label='Gauss Siedel')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('SOR')
    plt.grid(color='black', linestyle='--', linewidth=0.5)
    plt.show()







# Parametri del problema
nx = 100  # Numero di punti sulla griglia lungo la direzione x
ny = 100  # Numero di punti sulla griglia lungo la direzione y
lx = 100.0  # Lunghezza dello spazio lungo la direzione x
ly = 100.0  # Lunghezza dello spazio lungo la direzione y
num_levels = 4  # Numero di livelli nella gerarchia multigrid

    # Crea una funzione di prova per la densità di carica o la sorgente (es. una carica puntiforme nel centro)
def rho(x, y):
        x_center, y_center = lx / 2.0, ly / 2.0
        r = np.sqrt((x - x_center)**2 + (y - y_center)**2)
        
        return np.exp(-r**2)

    # Crea un'istanza del solver multigrid
solver = MultigridSolver2(nx, ny, lx, ly, num_levels)

    # Risolvi l'equazione di Poisson 2D utilizzando il metodo del ciclo V
phi = solver.solve_poisson_equation_2d(rho)

    # Stampa il campo scalare φ calcolato
print("Campo scalare φ:")
print(phi)



plt.imshow(phi, cmap='jet', origin='lower', extent=[0, lx, 0, ly])

plt.colorbar(label='Gauss Siedel')
plt.xlabel('x')
plt.ylabel('y')
plt.title('SOR')
plt.grid(color='black', linestyle='--', linewidth=0.5)
plt.show()
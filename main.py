import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
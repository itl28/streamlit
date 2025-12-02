import streamlit as st
import pandas as pd
import time as time
from maze_solver import (
    MAZE,
    START,
    END,
    solve_maze_bfs,
    solve_maze_dfs,
    solve_maze_astar
)

st.title("Visualizador de Algoritmo de Búsqueda en Laberinto")

# Función para renderizar el laberinto
def render_maze(maze, path=None):
    if path is None:
        path = []

    display_maze = []
    for r_idx, row in enumerate(maze):
        display_row = []
        for c_idx, col in enumerate(row):
            if (r_idx, c_idx) == START:
                display_row.append("🚀")  # Inicio
            elif (r_idx, c_idx) == END:
                display_row.append("🏁")  # Fin
            elif (r_idx, c_idx) in path:
                display_row.append("🔹")  # Camino resuelto
            elif col == 1:
                display_row.append("⬛")  # Muro
            else:
                display_row.append("⬜")  # Camino libre
        display_maze.append("".join(display_row))

    st.markdown("<br>".join(display_maze), unsafe_allow_html=True)


# Sidebar para controles
st.sidebar.header("Opciones")
algorithm = st.sidebar.selectbox(
    "Selecciona el algoritmo",
    ["BFS", "DFS", "A*"]
)
solve_button = st.sidebar.button("Resolver Laberinto")

# Mostrar laberinto inicial
render_maze(MAZE)

# Lógica de resolución
if solve_button:
    if algorithm == "BFS":
        solver = solve_maze_bfs
    elif algorithm == "DFS":
        solver = solve_maze_dfs
    else:  # "A*"
        solver = solve_maze_astar

    start_time = time.time()
    path = solver(MAZE, START, END)
    end_time = time.time()
    tiempo_ejecucion = end_time - start_time

    if path:
        st.success(f"¡Camino encontrado con {algorithm}!")
        render_maze(MAZE, path)
        st.write(f"Tiempo de ejecución: {tiempo_ejecucion:.5f} segundos")
        st.write(f"Longitud del camino: {len(path)} pasos")
    else:
        st.error(f"No se encontró un camino usando {algorithm}.")

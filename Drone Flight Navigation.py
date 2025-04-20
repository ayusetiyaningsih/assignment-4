import heapq
import math
import time

class Node:
    def __init__(self, position, parent=None, g=0, h=0):
        self.position = position
        self.parent = parent
        self.g = g
        self.h = h

    def f(self):
        return self.g + self.h

    def __lt__(self, other):
        return self.f() < other.f()

def heuristic(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def get_neighbors(pos, grid):
    directions = [(-1,0), (1,0), (0,-1), (0,1)]
    neighbors = []
    for dx, dy in directions:
        nx, ny = pos[0] + dx, pos[1] + dy
        if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]):
            if grid[nx][ny] != '#':
                neighbors.append((nx, ny))
    return neighbors

def elevation_cost(current, neighbor, grid):
    def val(pos):
        ch = grid[pos[0]][pos[1]]
        return int(ch) if ch.isdigit() else 1
    return abs(val(current) - val(neighbor)) + 1

def reconstruct_path(node):
    path = []
    while node:
        path.append(node.position)
        node = node.parent
    return path[::-1]

def a_star(grid, start, goal):
    start_time = time.time()
    open_set = []
    start_node = Node(start, g=0, h=heuristic(start, goal))
    heapq.heappush(open_set, start_node)
    visited = set()
    node_count = 0

    while open_set:
        current = heapq.heappop(open_set)
        node_count += 1

        if current.position == goal:
            return reconstruct_path(current), node_count, time.time() - start_time

        visited.add(current.position)

        for neighbor in get_neighbors(current.position, grid):
            if neighbor in visited:
                continue
            g = current.g + elevation_cost(current.position, neighbor, grid)
            h = heuristic(neighbor, goal)
            heapq.heappush(open_set, Node(neighbor, current, g, h))

    return [], node_count, time.time() - start_time

def gbfs(grid, start, goal):
    start_time = time.time()
    open_set = []
    start_node = Node(start, g=0, h=heuristic(start, goal))
    heapq.heappush(open_set, start_node)
    visited = set()
    node_count = 0

    while open_set:
        current = heapq.heappop(open_set)
        node_count += 1

        if current.position == goal:
            return reconstruct_path(current), node_count, time.time() - start_time

        visited.add(current.position)

        for neighbor in get_neighbors(current.position, grid):
            if neighbor in visited:
                continue
            h = heuristic(neighbor, goal)
            heapq.heappush(open_set, Node(neighbor, current, 0, h))

    return [], node_count, time.time() - start_time

def print_grid(grid, path):
    grid_copy = [row[:] for row in grid]
    for x, y in path:
        if grid_copy[x][y] not in ['S', 'G']:
            grid_copy[x][y] = '*'
    for row in grid_copy:
        print(' '.join(row))

# Contoh Grid Terrain
# Elevasi 1–9, S = Start, G = Goal, # = zona terlarang
terrain_grid = [
    ['S', '1', '2', '3', 'G'],
    ['#', '#', '2', '#', '3'],
    ['3', '3', '3', '3', '4'],
    ['4', '#', '5', '#', '5'],
    ['5', '6', '7', '8', '9'],
]

# Cari posisi start dan goal
def find_symbol(grid, symbol):
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == symbol:
                return (i, j)
    return None

start = find_symbol(terrain_grid, 'S')
goal = find_symbol(terrain_grid, 'G')

# Run A*
print("A* Search:")
a_path, a_nodes, a_time = a_star(terrain_grid, start, goal)
print_grid(terrain_grid, a_path)
print(f"Nodes Visited: {a_nodes}")
print(f"Execution Time: {a_time:.6f} seconds\n")

# Run GBFS
print("Greedy Best-First Search:")
g_path, g_nodes, g_time = gbfs(terrain_grid, start, goal)
print_grid(terrain_grid, g_path)
print(f"Nodes Visited: {g_nodes}")
print(f"Execution Time: {g_time:.6f} seconds")

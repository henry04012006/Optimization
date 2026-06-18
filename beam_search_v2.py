import heapq
import sys

def solve_beam_pure_depot(N, K, d_matrix, beam_width=150, branch_limit=5):
    initial_routes = [[0] for _ in range(K)]
    initial_lengths = [0] * K
    initial_unvisited = list(range(1, N + 1))
    
    beam = [(0, initial_routes, initial_lengths, initial_unvisited)]
    
    while beam[0][3]:
        candidates = []
        for current_max, routes, lengths, unvisited in beam:
            
            target_idx = 0
            min_proj_len = float('inf')
            for i in range(K):
                proj_l = lengths[i] + d_matrix[routes[i][-1]][0]
                if proj_l < min_proj_len:
                    min_proj_len = proj_l
                    target_idx = i
                    
            last_node = routes[target_idx][-1]
            
            unv_sorted = sorted(unvisited, key=lambda x: d_matrix[last_node][x])
            
            for next_node in unv_sorted[:branch_limit]:
                new_routes = [r[:] for r in routes]
                new_lengths = lengths[:]
                
                new_routes[target_idx].append(next_node)
                new_lengths[target_idx] += d_matrix[last_node][next_node]
                
                new_unvisited = [n for n in unvisited if n != next_node]
                
                new_max = 0
                for i in range(K):
                    l = new_lengths[i] + d_matrix[new_routes[i][-1]][0]
                    if l > new_max:
                        new_max = l
                
                candidates.append((new_max, new_routes, new_lengths, new_unvisited))
        
        beam = heapq.nsmallest(beam_width, candidates, key=lambda x: x[0])
        
    final_solution = beam[0][1]
    
    print(K)
    for route in final_solution:
        print(len(route))
        print(*(route))

if __name__ == "__main__":
    input_data = sys.stdin.read().split()
    if input_data:
        N = int(input_data[0])
        K = int(input_data[1])
        d_matrix = []
        idx = 2
        for i in range(N + 1):
            row = []
            for j in range(N + 1):
                row.append(int(input_data[idx]))
                idx += 1
            d_matrix.append(row)
        solve_beam_pure_depot(N, K, d_matrix)

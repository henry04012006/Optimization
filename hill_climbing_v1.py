import sys

def read_input():
    numbers = list(map(int, sys.stdin.buffer.read().split()))
    if not numbers:
        return 0, 0, []
    
    number_of_points = numbers[0]
    number_of_postmen = numbers[1]
    
    matrix_size = number_of_points + 1
    distances = []
    position = 2
    for _ in range(matrix_size):
        distances.append(numbers[position : position + matrix_size])
        position += matrix_size
        
    return number_of_points, number_of_postmen, distances

def naive_initialization(number_of_points, number_of_postmen, distances):
    routes = [[0] for _ in range(number_of_postmen)]
    lengths = [0] * number_of_postmen
    
    for i in range(1, number_of_points + 1):
        target_k = (i - 1) % number_of_postmen
        last_node = routes[target_k][-1]
        routes[target_k].append(i)
        lengths[target_k] += distances[last_node][i]
        
    return routes, lengths

def get_system_costs(routes, lengths, distances, number_of_postmen):
    costs = [0] * number_of_postmen
    max_cost = 0
    max_k = -1
    for k in range(number_of_postmen):
        costs[k] = lengths[k] + distances[routes[k][-1]][0]
        if costs[k] > max_cost:
            max_cost = costs[k]
            max_k = k
    return costs, max_cost, max_k

def solve_hill_climbing(number_of_points, number_of_postmen, distances):
    routes, lengths = naive_initialization(number_of_points, number_of_postmen, distances)
    
    improved = True
    while improved:
        improved = False
        costs, current_max, max_k = get_system_costs(routes, lengths, distances, number_of_postmen)
        
        for i in range(1, len(routes[max_k])):
            node = routes[max_k][i]
            prev_max = routes[max_k][i-1]
            next_max = routes[max_k][i+1] if i < len(routes[max_k]) - 1 else 0
            
            delta_max = distances[prev_max][next_max] - (distances[prev_max][node] + distances[node][next_max])
            new_len_max = lengths[max_k] + delta_max
            last_max = routes[max_k][-1] if i < len(routes[max_k]) - 1 else routes[max_k][-2]
            new_cost_max = new_len_max + distances[last_max][0]
            
            for k2 in range(number_of_postmen):
                if k2 == max_k: continue
                
                for j in range(1, len(routes[k2]) + 1):
                    prev_2 = routes[k2][j-1]
                    next_2 = routes[k2][j] if j < len(routes[k2]) else 0
                    
                    delta_2 = distances[prev_2][node] + distances[node][next_2] - distances[prev_2][next_2]
                    new_len_2 = lengths[k2] + delta_2
                    last_2 = routes[k2][-1] if j < len(routes[k2]) else node
                    new_cost_2 = new_len_2 + distances[last_2][0]
                    
                    if new_cost_max < current_max and new_cost_2 < current_max:
                      
                        proposed_max = max(new_cost_max, new_cost_2)
                        for x in range(number_of_postmen):
                            if x != max_k and x != k2 and costs[x] > proposed_max:
                                proposed_max = costs[x]
                        
                        if proposed_max < current_max:
                            routes[max_k].pop(i)
                            routes[k2].insert(j, node)
                            lengths[max_k] = new_len_max
                            lengths[k2] = new_len_2
                            improved = True
                            break
                if improved: break
            if improved: break
            
        if improved: continue

        for i in range(1, len(routes[max_k])):
            node_max = routes[max_k][i]
            prev_max = routes[max_k][i-1]
            next_max = routes[max_k][i+1] if i < len(routes[max_k]) - 1 else 0
            
            for k2 in range(number_of_postmen):
                if k2 == max_k: continue
                
                for j in range(1, len(routes[k2])):
                    node_2 = routes[k2][j]
                    prev_2 = routes[k2][j-1]
                    next_2 = routes[k2][j+1] if j < len(routes[k2]) - 1 else 0
                    
                    delta_max = (distances[prev_max][node_2] + distances[node_2][next_max]) - (distances[prev_max][node_max] + distances[node_max][next_max])
                    new_len_max = lengths[max_k] + delta_max
                    last_max = routes[max_k][-1] if i < len(routes[max_k]) - 1 else node_2
                    new_cost_max = new_len_max + distances[last_max][0]
                    
                    delta_2 = (distances[prev_2][node_max] + distances[node_max][next_2]) - (distances[prev_2][node_2] + distances[node_2][next_2])
                    new_len_2 = lengths[k2] + delta_2
                    last_2 = routes[k2][-1] if j < len(routes[k2]) - 1 else node_max
                    new_cost_2 = new_len_2 + distances[last_2][0]
                    
                    if new_cost_max < current_max and new_cost_2 < current_max:
                        proposed_max = max(new_cost_max, new_cost_2)
                        for x in range(number_of_postmen):
                            if x != max_k and x != k2 and costs[x] > proposed_max:
                                proposed_max = costs[x]
                                
                        if proposed_max < current_max:
                            routes[max_k][i], routes[k2][j] = routes[k2][j], routes[max_k][i]
                            lengths[max_k] = new_len_max
                            lengths[k2] = new_len_2
                            improved = True
                            break
                if improved: break
            if improved: break

    return routes

def print_solution(routes):
    output_lines = [str(len(routes))]
    for route in routes:
        output_lines.append(str(len(route)))
        output_lines.append(" ".join(map(str, route)))
    sys.stdout.write("\n".join(output_lines) + "\n")

def main():
    n, k, dist = read_input()
    if n > 0:
        routes = solve_hill_climbing(n, k, dist)
        print_solution(routes)

if __name__ == "__main__":
    main()

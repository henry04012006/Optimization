import sys

def solve():
    input_data = sys.stdin.read().split()
    if not input_data:
        return

    N = int(input_data[0])
    K = int(input_data[1])

    dist = []
    idx = 2
    for _ in range(N + 1):
        row = []
        for _ in range(N + 1):
            row.append(int(input_data[idx]))
            idx += 1
        dist.append(row)

    routes = [[0] for _ in range(K)]
    lengths = [0] * K
    
    unvisited = set(range(1, N + 1))
    while unvisited:
        best_node, best_k, min_increase = -1, -1, float('inf')

        for v in unvisited:
            for k in range(K):
                last_node = routes[k][-1]
                increase = dist[last_node][v] + dist[v][0] - dist[last_node][0]
                if increase < min_increase:
                    min_increase = increase
                    best_node = v
                    best_k = k

        lengths[best_k] += dist[routes[best_k][-1]][best_node]
        routes[best_k].append(best_node)
        unvisited.remove(best_node)

    def get_route_cost(k, r_list, l_list):
        return l_list[k] + dist[r_list[k][-1]][0]

    improved = True
    while improved:
        improved = False
        current_costs = [get_route_cost(k, routes, lengths) for k in range(K)]
        current_max_cost = max(current_costs)
        
        for k1 in range(K):
            if improved: break
            for i in range(1, len(routes[k1])):
                if improved: break
                node_to_move = routes[k1][i]
                prev_1 = routes[k1][i-1]
                next_1 = routes[k1][i+1] if i < len(routes[k1]) - 1 else 0
                
                delta_1 = dist[prev_1][next_1] - (dist[prev_1][node_to_move] + dist[node_to_move][next_1])
                new_l1 = lengths[k1] + delta_1
                last_1 = routes[k1][-1] if i < len(routes[k1]) - 1 else routes[k1][-2]
                new_cost_1 = new_l1 + dist[last_1][0]
                
                for k2 in range(K):
                    if k1 == k2: continue
                    if improved: break
                        
                    for j in range(1, len(routes[k2]) + 1):
                        prev_2 = routes[k2][j-1]
                        next_2 = routes[k2][j] if j < len(routes[k2]) else 0
                      
                        delta_2 = dist[prev_2][node_to_move] + dist[node_to_move][next_2] - dist[prev_2][next_2]
                        new_l2 = lengths[k2] + delta_2
                        last_2 = routes[k2][-1] if j < len(routes[k2]) else node_to_move
                        new_cost_2 = new_l2 + dist[last_2][0]
                        
                        new_max = max(new_cost_1, new_cost_2)
                        for k in range(K):
                            if k != k1 and k != k2:
                                if current_costs[k] > new_max:
                                    new_max = current_costs[k]
                        
                        if new_max < current_max_cost:
                            routes[k1].pop(i)
                            routes[k2].insert(j, node_to_move)
                            lengths[k1] = new_l1
                            lengths[k2] = new_l2
                            improved = True
                            break

        if improved: continue

        for k1 in range(K):
            if improved: break
            for i in range(1, len(routes[k1])):
                if improved: break
                for k2 in range(k1 + 1, K):
                    if improved: break
                    for j in range(1, len(routes[k2])):
                        node_1 = routes[k1][i]
                        prev_1 = routes[k1][i-1]
                        next_1 = routes[k1][i+1] if i < len(routes[k1]) - 1 else 0
                        
                        node_2 = routes[k2][j]
                        prev_2 = routes[k2][j-1]
                        next_2 = routes[k2][j+1] if j < len(routes[k2]) - 1 else 0
                        
                        delta_1 = (dist[prev_1][node_2] + dist[node_2][next_1]) - (dist[prev_1][node_1] + dist[node_1][next_1])
                        new_l1 = lengths[k1] + delta_1
                        last_1 = routes[k1][-1] if i < len(routes[k1]) - 1 else node_2
                        new_cost_1 = new_l1 + dist[last_1][0]
                        
                        delta_2 = (dist[prev_2][node_1] + dist[node_1][next_2]) - (dist[prev_2][node_2] + dist[node_2][next_2])
                        new_l2 = lengths[k2] + delta_2
                        last_2 = routes[k2][-1] if j < len(routes[k2]) - 1 else node_1
                        new_cost_2 = new_l2 + dist[last_2][0]
                        
                        new_max = max(new_cost_1, new_cost_2)
                        for k in range(K):
                            if k != k1 and k != k2:
                                if current_costs[k] > new_max:
                                    new_max = current_costs[k]
                        
                        if new_max < current_max_cost:
                            routes[k1][i], routes[k2][j] = routes[k2][j], routes[k1][i]
                            lengths[k1] = new_l1
                            lengths[k2] = new_l2
                            improved = True
                            break

    print(K)
    for route in routes:
        print(len(route))
        print(*(route))

if __name__ == "__main__":
    solve()

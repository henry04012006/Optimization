import sys

def read_input():
    data = list(map(int, sys.stdin.buffer.read().split()))
    if not data: return 0, 0, []
    
    n = data[0]
    k = data[1]
    
    dist = []
    pos = 2
    mat_size = n + 1
    for _ in range(mat_size):
        dist.append(data[pos : pos + mat_size])
        pos += mat_size

    for i in range(n+1):
        dist[i][0] = 0
        
    return n, k, dist

def init_routes(n, k, dist):
    routes = [[0] for _ in range(k)]
    route_lengths = [0] * k
    unvisited = list(range(1, n + 1))
    
    while unvisited:
        best_route = 0
        best_idx = 0
        best_node = unvisited[0]
        best_val = float('inf')
        
        for v in range(k):
            last_node = routes[v][-1]
            cur_len = route_lengths[v]
            
            for idx, node in enumerate(unvisited):
                val = cur_len + dist[last_node][node]
                if val < best_val:
                    best_val = val
                    best_route = v
                    best_idx = idx
                    best_node = node
                    
        routes[best_route].append(best_node)
        route_lengths[best_route] += dist[routes[best_route][-2]][best_node]
        unvisited.pop(best_idx)
        
    return [r[1:] for r in routes]

def solve(n, k, dist):
    routes = init_routes(n, k, dist)
    
    lengths = [0] * k
    for v in range(k):
        if routes[v]:
            c = dist[0][routes[v][0]]
            for i in range(len(routes[v])-1):
                c += dist[routes[v][i]][routes[v][i+1]]
            lengths[v] = c

    def get_cost(v):
        if not routes[v]: return 0
        return lengths[v] + dist[routes[v][-1]][0]

    improved = True
    while improved:
        improved = False
        costs = [get_cost(v) for v in range(k)]
        max_c = max(costs)
        sum_c = sum(costs)
        
        for v in range(k):
            r = routes[v]
            sz = len(r)
            if sz < 2: continue
            for i in range(sz - 1):
                for j in range(i + 1, sz):
                    prev_i = r[i-1] if i > 0 else 0
                    node_i = r[i]
                    node_j = r[j]
                    next_j = r[j+1] if j < sz - 1 else 0
                    
                    delta = (dist[prev_i][node_j] + dist[node_i][next_j]) - \
                            (dist[prev_i][node_i] + dist[node_j][next_j])
                    
                    if delta < 0:
                        r[i:j+1] = reversed(r[i:j+1])
                        lengths[v] += delta
                        costs[v] += delta
                        max_c = max(costs)
                        sum_c += delta
                        improved = True
                        break
                if improved: break
            if improved: break
        if improved: continue

        bottleneck_v = costs.index(max_c)
        target_vehicles = list(range(k))
        target_vehicles.remove(bottleneck_v)
        target_vehicles.sort(key=lambda x: costs[x])
        
        b_route = routes[bottleneck_v]
        b_sz = len(b_route)
        
        for i in range(b_sz):
            node = b_route[i]
            prev_b = b_route[i-1] if i > 0 else 0
            next_b = b_route[i+1] if i < b_sz - 1 else 0
            
            delta_b = dist[prev_b][next_b] - (dist[prev_b][node] + dist[node][next_b])
            new_cost_b = costs[bottleneck_v] + delta_b
            
            for v in target_vehicles:
                t_route = routes[v]
                t_sz = len(t_route)
                
                for j in range(t_sz + 1):
                    prev_t = t_route[j-1] if j > 0 else 0
                    next_t = t_route[j] if j < t_sz else 0
                    
                    delta_t = dist[prev_t][node] + dist[node][next_t] - dist[prev_t][next_t]
                    new_cost_t = costs[v] + delta_t
                    
                    temp_max = 0
                    for x in range(k):
                        if x == bottleneck_v: c_x = new_cost_b
                        elif x == v: c_x = new_cost_t
                        else: c_x = costs[x]
                        if c_x > temp_max: temp_max = c_x
                        
                    new_sum_c = sum_c + delta_b + delta_t
                    
                    if temp_max < max_c or (temp_max == max_c and new_sum_c < sum_c):
                        t_route.insert(j, node)
                        b_route.pop(i)
                        lengths[bottleneck_v] += delta_b
                        lengths[v] += delta_t
                        costs[bottleneck_v] = new_cost_b
                        costs[v] = new_cost_t
                        improved = True
                        break
                if improved: break
            if improved: break
        if improved: continue

        for i in range(b_sz):
            node_b = b_route[i]
            prev_b = b_route[i-1] if i > 0 else 0
            next_b = b_route[i+1] if i < b_sz - 1 else 0
            
            for v in target_vehicles:
                t_route = routes[v]
                t_sz = len(t_route)
                
                for j in range(t_sz):
                    node_t = t_route[j]
                    prev_t = t_route[j-1] if j > 0 else 0
                    next_t = t_route[j+1] if j < t_sz - 1 else 0
                    
                    delta_b = (dist[prev_b][node_t] + dist[node_t][next_b]) - (dist[prev_b][node_b] + dist[node_b][next_b])
                    delta_t = (dist[prev_t][node_b] + dist[node_b][next_t]) - (dist[prev_t][node_t] + dist[node_t][next_t])
                    
                    new_cost_b = costs[bottleneck_v] + delta_b
                    new_cost_t = costs[v] + delta_t
                    
                    temp_max = 0
                    for x in range(k):
                        if x == bottleneck_v: c_x = new_cost_b
                        elif x == v: c_x = new_cost_t
                        else: c_x = costs[x]
                        if c_x > temp_max: temp_max = c_x
                        
                    new_sum_c = sum_c + delta_b + delta_t
                    
                    if temp_max < max_c or (temp_max == max_c and new_sum_c < sum_c):
                        b_route[i] = node_t
                        t_route[j] = node_b
                        lengths[bottleneck_v] += delta_b
                        lengths[v] += delta_t
                        costs[bottleneck_v] = new_cost_b
                        costs[v] = new_cost_t
                        improved = True
                        break
                if improved: break
            if improved: break

    return routes

def print_res(routes):
    out = [str(len(routes))]
    for r in routes:
        final_r = [0] + r
        out.append(str(len(final_r)))
        out.append(" ".join(map(str, final_r)))
    sys.stdout.write("\n".join(out) + "\n")

def main():
    n, k, dist = read_input()
    if n > 0:
        ans = solve(n, k, dist)
        print_res(ans)

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()

import sys

def read_input():
    data = list(map(int, sys.stdin.buffer.read().split()))
    if not data:
        return 0, 0, []
    
    n = data[0]
    k = data[1]
    
    mat_size = n + 1
    dist = []
    pos = 2
    for _ in range(mat_size):
        dist.append(data[pos : pos + mat_size])
        pos += mat_size
        
    return n, k, dist

def solve_beam(n, k, dist):
    BW = 15 
    
    nodes = list(range(1, n + 1))
    nodes.sort(key=lambda x: dist[0][x], reverse=True)
    
    init_routes = [[0] for _ in range(k)]
    init_lengths = [0] * k
    beam = [(0, 0, init_routes, init_lengths)]
    
    for node in nodes:
        next_beam = []
        for _, _, routes, lengths in beam:
            for v in range(k):
                prev_node = routes[v][-1]
                added_d = dist[prev_node][node]
                new_len_v = lengths[v] + added_d
                
                new_max = 0
                new_tot = 0
                
                for i in range(k):
                    if i == v:
                        c = new_len_v + dist[node][0]
                    else:
                        c = lengths[i] + dist[routes[i][-1]][0]
                        
                    if c > new_max:
                        new_max = c
                    new_tot += c
                
                new_routes = [r[:] for r in routes]
                new_routes[v].append(node)
                new_lengths = lengths[:]
                new_lengths[v] = new_len_v
                
                next_beam.append((new_max, new_tot, new_routes, new_lengths))
        
        next_beam.sort(key=lambda x: (x[0], x[1]))
        beam = next_beam[:BW]
        
    return beam[0][2]

def print_res(routes):
    out = [str(len(routes))]
    for r in routes:
        out.append(str(len(r)))
        out.append(" ".join(map(str, r)))
    sys.stdout.write("\n".join(out) + "\n")

def main():
    n, k, dist = read_input()
    if n > 0:
        ans = solve_beam(n, k, dist)
        print_res(ans)

if __name__ == "__main__":
    main()

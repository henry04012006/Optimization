import sys
import time
import random
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional

INF = 10 ** 18
N = 0
K = 0
SIZE = 0
D: List[int] = []          
d: List[List[int]] = []    
rng = random.Random(time.time_ns())

config = {
    "time_limit": 29,
}


@dataclass
class Individual:
    routes: List[List[int]] = field(default_factory=list)
    maxCost: int = 0
    totalCost: int = 0


# ---------------- input / config ----------------
def input_data() -> Tuple[int, int, List[List[int]], List[int]]:
    data = list(map(int, sys.stdin.buffer.read().split()))
    if len(data) < 2:
        return 0, 0, [], []

    n, k = data[0], data[1]
    need = 2 + (n + 1) * (n + 1)
    if n <= 0 or k <= 0 or len(data) < need:
        return 0, 0, [], []

    ptr = 2
    dist: List[List[int]] = []
    flat: List[int] = []
    for _ in range(n + 1):
        row = data[ptr:ptr + n + 1]
        ptr += n + 1
        row[0] = 0
        dist.append(row)
        flat.extend(row)

    return n, k, dist, flat


def get_config(n: int, k: int) -> Dict[str, Any]:
    if n <= 250:
        return {
            "mode": "small",
            "time_limit": config["time_limit"],
            "main_time_ratio": 0.995,
            "repair_route_limit": k,

            "polish_passes": 45,
            "final_polish_passes": 75,
            "relocate_sample": 42,
            "relocate_dest": min(10, max(1, k - 1)),
            "swap_sample": 22,
            "swap_dest": min(7, max(1, k - 1)),
            "swap_target": 9,
            "intra_sample": 16,
            "intra_pos": 16,

            "ruin_min": max(2, n // 100),
            "ruin_max": max(7, n // 16),
            "ruin_prob": 0.18,
            "t0_ratio": 0.018,
            "cooling": 0.9965,
            "min_temp": 1e-6,
            "total_lambda": 1e-4,
            "restart_no_improve": 150,
        }

    if n <= 600:
        return {
            "mode": "medium",
            "time_limit": config["time_limit"],
            "main_time_ratio": 0.995,
            "repair_route_limit": min(7, k),

            "polish_passes": 28,
            "final_polish_passes": 45,
            "relocate_sample": 24,
            "relocate_dest": min(6, max(1, k - 1)),
            "swap_sample": 12,
            "swap_dest": min(5, max(1, k - 1)),
            "swap_target": 6,
            "intra_sample": 8,
            "intra_pos": 8,

            "ruin_min": max(3, n // 180),
            "ruin_max": max(9, n // 30),
            "ruin_prob": 0.20,
            "t0_ratio": 0.022,
            "cooling": 0.9975,
            "min_temp": 1e-6,
            "total_lambda": 1e-4,
            "restart_no_improve": 115,
        }

    return {
        "mode": "large",
        "time_limit": config["time_limit"],
        "main_time_ratio": 0.995,
        "repair_route_limit": min(5, k),

        "polish_passes": 16,
        "final_polish_passes": 25,
        "relocate_sample": 10,
        "relocate_dest": min(4, max(1, k - 1)),
        "swap_sample": 6,
        "swap_dest": min(3, max(1, k - 1)),
        "swap_target": 4,
        "intra_sample": 5,
        "intra_pos": 5,

        "ruin_min": 3,
        "ruin_max": max(10, n // 70),
        "ruin_prob": 0.24,
        "t0_ratio": 0.026,
        "cooling": 0.9983,
        "min_temp": 1e-6,
        "total_lambda": 1e-4,
        "restart_no_improve": 85,
    }


# ---------------- basic evaluation ----------------
def dist(a: int, b: int) -> int:
    return D[a * SIZE + b]


def route_cost(route: List[int]) -> int:
    flat = D
    s = SIZE
    cost = 0
    for i in range(len(route) - 1):
        cost += flat[route[i] * s + route[i + 1]]
    return cost


def get_costs(routes: List[List[int]]) -> List[int]:
    return [route_cost(route) for route in routes]


def better(a: Individual, b: Individual) -> bool:
    return (a.maxCost, a.totalCost) < (b.maxCost, b.totalCost)


def clone(ind: Individual) -> Individual:
    return Individual([r[:] for r in ind.routes], ind.maxCost, ind.totalCost)


def evaluate_clean(routes: List[List[int]]) -> Individual:
    costs = get_costs(routes)
    mx = max(costs) if costs else 0
    total = sum(costs)
    return Individual([r[:] for r in routes], mx, total)


def remove_delta(route: List[int], idx: int) -> int:
    flat = D
    s = SIZE
    p = route[idx - 1]
    x = route[idx]
    q = route[idx + 1]
    return flat[p * s + q] - flat[p * s + x] - flat[x * s + q]


def insert_delta(route: List[int], pos: int, x: int) -> int:
    flat = D
    s = SIZE
    a = route[pos - 1]
    b = route[pos]
    return flat[a * s + x] + flat[x * s + b] - flat[a * s + b]


# ---------------- repair / insertion ----------------
def best_insertion(routes: List[List[int]], costs: List[int], x: int,
                   route_limit: Optional[int] = None) -> Tuple[int, int, int, int, int]:
    flat = D
    s = SIZE
    old_total = sum(costs)

    if route_limit is not None and route_limit < K:
        order = sorted(range(K), key=costs.__getitem__)[:route_limit]
    else:
        order = range(K)

    max1 = -1
    max2 = -1
    argmax = -1
    for i, c in enumerate(costs):
        if c > max1:
            max2 = max1
            max1 = c
            argmax = i
        elif c > max2:
            max2 = c

    best_new_max = INF
    best_new_total = INF
    best_delta = 0
    best_k = 0
    best_pos = 1

    for k in order:
        other_max = max2 if k == argmax else max1
        if other_max < 0:
            other_max = 0
        route = routes[k]
        base_cost = costs[k]
        for pos in range(1, len(route)):
            a = route[pos - 1]
            b = route[pos]
            delta = flat[a * s + x] + flat[x * s + b] - flat[a * s + b]
            new_cost = base_cost + delta
            new_max = other_max if other_max >= new_cost else new_cost
            new_total = old_total + delta
            if new_max < best_new_max or (new_max == best_new_max and new_total < best_new_total):
                best_new_max = new_max
                best_new_total = new_total
                best_delta = delta
                best_k = k
                best_pos = pos

    return best_new_max, best_new_total, best_delta, best_k, best_pos


# ---------------- repair validity ----------------
def repair_routes(routes: List[List[int]]) -> List[List[int]]:
    if K <= 0:
        return []

    routes = [r[:] for r in routes[:K]]
    while len(routes) < K:
        routes.append([0, 0])

    clean: List[List[int]] = []
    seen = [False] * (N + 1)

    for route in routes:
        nr = [0]
        for x in route:
            if 1 <= x <= N and not seen[x]:
                seen[x] = True
                nr.append(x)
        nr.append(0)
        clean.append(nr)

    missing = [x for x in range(1, N + 1) if not seen[x]]
    if not missing:
        return clean

    costs = get_costs(clean)
    for x in missing:
        _, _, delta, k, pos = best_insertion(clean, costs, x, K)
        clean[k].insert(pos, x)
        costs[k] += delta

    return clean


def make_individual(routes: List[List[int]]) -> Individual:
    return evaluate_clean(repair_routes(routes))


# ---------------- greedy initial solution only ----------------
def greedy_initial() -> Individual:
    routes = [[0, 0] for _ in range(K)]
    costs = [0] * K
    nodes = list(range(1, N + 1))

    nodes.sort(key=lambda x: D[x], reverse=True)

    for x in nodes:
        _, _, delta, k, pos = best_insertion(routes, costs, x, K)
        routes[k].insert(pos, x)
        costs[k] += delta

    return evaluate_clean(routes)


# ---------------- guided local moves ----------------
def max_excluding_two(costs: List[int], a: int, b: int) -> int:
    best = 0
    for i, c in enumerate(costs):
        if i != a and i != b and c > best:
            best = c
    return best


def candidate_indices_from_route(route: List[int], limit: int) -> List[int]:
    if len(route) <= 2:
        return []
    indices = list(range(1, len(route) - 1))
    if len(indices) <= limit:
        return indices

    items = [(remove_delta(route, i), i) for i in indices]
    items.sort()              
    keep = max(1, limit // 2)
    chosen = [i for _delta, i in items[:keep]]
    chosen_set = set(chosen)
    rest = [i for i in indices if i not in chosen_set]
    rng.shuffle(rest)
    return chosen + rest[:limit - keep]


def guided_relocate_from_longest(routes: List[List[int]], costs: List[int],
                                 cfg: Dict[str, Any], require_improve: bool) -> bool:
    if K <= 1:
        return False

    longest = max(range(K), key=costs.__getitem__)
    if len(routes[longest]) <= 2:
        return False

    old_max = max(costs)
    old_total = sum(costs)
    best_key = (old_max, old_total) if require_improve else None
    best_move = None

    source_indices = candidate_indices_from_route(routes[longest], cfg["relocate_sample"])
    dest_routes = [k for k in range(K) if k != longest]
    dest_routes.sort(key=costs.__getitem__)
    dest_routes = dest_routes[:cfg["relocate_dest"]]

    for i in source_indices:
        if i >= len(routes[longest]) - 1:
            continue
        x = routes[longest][i]
        rem = remove_delta(routes[longest], i)
        new_long = costs[longest] + rem

        for k in dest_routes:
            other_max = max_excluding_two(costs, longest, k)
            r = routes[k]
            ck = costs[k]
            for pos in range(1, len(r)):
                ins = insert_delta(r, pos, x)
                new_k = ck + ins
                key = (max(new_long, new_k, other_max), old_total + rem + ins)
                if best_key is None or key < best_key:
                    best_key = key
                    best_move = (i, k, pos, rem, ins)

    if best_move is None:
        return False

    i, k, pos, rem, ins = best_move
    x = routes[longest].pop(i)
    routes[k].insert(pos, x)
    costs[longest] += rem
    costs[k] += ins
    return True


def guided_swap_from_longest(routes: List[List[int]], costs: List[int],
                             cfg: Dict[str, Any], require_improve: bool) -> bool:
    if K <= 1:
        return False

    flat = D
    s = SIZE
    longest = max(range(K), key=costs.__getitem__)
    if len(routes[longest]) <= 2:
        return False

    old_max = max(costs)
    old_total = sum(costs)
    best_key = (old_max, old_total) if require_improve else None
    best_move = None

    source_indices = candidate_indices_from_route(routes[longest], cfg["swap_sample"])
    dest_routes = [k for k in range(K) if k != longest and len(routes[k]) > 2]
    dest_routes.sort(key=costs.__getitem__)
    dest_routes = dest_routes[:cfg["swap_dest"]]

    for i in source_indices:
        if i >= len(routes[longest]) - 1:
            continue
        r_long = routes[longest]
        x = r_long[i]
        px = r_long[i - 1]
        nx = r_long[i + 1]

        for k in dest_routes:
            r = routes[k]
            target_indices = list(range(1, len(r) - 1))
            if len(target_indices) > cfg["swap_target"]:
                target_indices = rng.sample(target_indices, cfg["swap_target"])

            other_max = max_excluding_two(costs, longest, k)
            for j in target_indices:
                y = r[j]
                py = r[j - 1]
                ny = r[j + 1]

                delta_long = flat[px * s + y] + flat[y * s + nx] - flat[px * s + x] - flat[x * s + nx]
                delta_k = flat[py * s + x] + flat[x * s + ny] - flat[py * s + y] - flat[y * s + ny]
                new_long = costs[longest] + delta_long
                new_k = costs[k] + delta_k
                key = (max(new_long, new_k, other_max), old_total + delta_long + delta_k)

                if best_key is None or key < best_key:
                    best_key = key
                    best_move = (i, k, j, delta_long, delta_k)

    if best_move is None:
        return False

    i, k, j, delta_long, delta_k = best_move
    routes[longest][i], routes[k][j] = routes[k][j], routes[longest][i]
    costs[longest] += delta_long
    costs[k] += delta_k
    return True


def guided_intra_reinsert_longest(routes: List[List[int]], costs: List[int],
                                  cfg: Dict[str, Any], require_improve: bool) -> bool:
    longest = max(range(K), key=costs.__getitem__)
    route = routes[longest]
    if len(route) <= 4:
        return False

    old_cost = costs[longest]
    old_max = max(costs)
    old_total = sum(costs)
    other_max = max((costs[k] for k in range(K) if k != longest), default=0)
    best_key = (old_max, old_total) if require_improve else None
    best_route = None
    best_cost = 0

    node_indices = list(range(1, len(route) - 1))
    if len(node_indices) > cfg["intra_sample"]:
        node_indices = rng.sample(node_indices, cfg["intra_sample"])

    for i in node_indices:
        pos_indices = list(range(1, len(route)))
        if len(pos_indices) > cfg["intra_pos"]:
            pos_indices = rng.sample(pos_indices, cfg["intra_pos"])

        for pos in pos_indices:
            if pos == i or pos == i + 1:
                continue
            tmp = route[:]
            node = tmp.pop(i)
            insert_pos = pos - 1 if pos > i else pos
            tmp.insert(insert_pos, node)
            new_cost = route_cost(tmp)
            key = (max(other_max, new_cost), old_total - old_cost + new_cost)
            if best_key is None or key < best_key:
                best_key = key
                best_route = tmp
                best_cost = new_cost

    if best_route is None:
        return False

    routes[longest] = best_route
    costs[longest] = best_cost
    return True


def greedy_polish(ind: Individual, cfg: Dict[str, Any], end_time: float, max_passes: int) -> Individual:
    routes = [r[:] for r in ind.routes]
    costs = get_costs(routes)
    pc = time.perf_counter

    for _ in range(max_passes):
        if pc() >= end_time:
            break
        if guided_relocate_from_longest(routes, costs, cfg, True):
            continue
        if guided_swap_from_longest(routes, costs, cfg, True):
            continue
        if guided_intra_reinsert_longest(routes, costs, cfg, True):
            continue
        break

    return evaluate_clean(routes)


# ---------------- ruin-recreate move ----------------
def repair_minmax(routes: List[List[int]], removed: List[int], route_limit: Optional[int] = None) -> None:
    costs = get_costs(routes)
    nodes = removed[:]
    nodes.sort(key=lambda x: D[x], reverse=True)

    for x in nodes:
        _, _, delta, k, pos = best_insertion(routes, costs, x, route_limit)
        routes[k].insert(pos, x)
        costs[k] += delta


def remove_from_longest_routes(routes: List[List[int]], q: int) -> List[int]:
    costs = get_costs(routes)
    removed: List[int] = []
    flat = D
    s = SIZE

    q = min(q, N)
    while len(removed) < q:
        longest = max(range(K), key=costs.__getitem__)
        route = routes[longest]
        if len(route) <= 2:
            non_empty = [k for k in range(K) if len(routes[k]) > 2]
            if not non_empty:
                break
            longest = max(non_empty, key=costs.__getitem__)
            route = routes[longest]

        candidates = []
        for i in range(1, len(route) - 1):
            p = route[i - 1]
            x = route[i]
            nxt = route[i + 1]
            rem = flat[p * s + nxt] - flat[p * s + x] - flat[x * s + nxt]
            candidates.append((rem, i, x))

        if not candidates:
            break

        candidates.sort()
        top = candidates[:min(3, len(candidates))]
        rem, i, x = rng.choice(top)
        route.pop(i)
        costs[longest] += rem
        removed.append(x)

    return removed


def candidate_guided_move(cur: Individual, cfg: Dict[str, Any]) -> Optional[Individual]:
    routes = [r[:] for r in cur.routes]
    costs = get_costs(routes)
    p = rng.random()

    if p < 0.48:
        changed = guided_relocate_from_longest(routes, costs, cfg, False)
    elif p < 0.76:
        changed = guided_swap_from_longest(routes, costs, cfg, False)
    else:
        changed = guided_intra_reinsert_longest(routes, costs, cfg, False)

    if not changed:
        return None
    return evaluate_clean(routes)


def candidate_ruin_recreate(cur: Individual, cfg: Dict[str, Any], no_improve: int) -> Optional[Individual]:
    routes = [r[:] for r in cur.routes]

    q_low = max(1, min(cfg["ruin_min"], N))
    q_high = max(q_low, min(cfg["ruin_max"], N))
    if no_improve > cfg["restart_no_improve"] // 2:
        q_high = min(N, int(q_high * 1.4) + 1)
    q = rng.randint(q_low, q_high)

    removed = remove_from_longest_routes(routes, q)
    if not removed:
        return None

    repair_minmax(routes, removed, cfg["repair_route_limit"])
    return evaluate_clean(routes)


def propose_candidate(cur: Individual, cfg: Dict[str, Any], no_improve: int) -> Optional[Individual]:
    if rng.random() > cfg["ruin_prob"]:
        cand = candidate_guided_move(cur, cfg)
        if cand is not None:
            return cand
    return candidate_ruin_recreate(cur, cfg, no_improve)


# ---------------- simulated annealing acceptance ----------------
def acceptance_delta(cand: Individual, cur: Individual, cfg: Dict[str, Any]) -> float:
    dmax = cand.maxCost - cur.maxCost
    dtotal = cand.totalCost - cur.totalCost
    if dmax < 0 or (dmax == 0 and dtotal <= 0):
        return -1.0
    return max(0.0, float(dmax)) + cfg["total_lambda"] * max(0.0, float(dtotal)) / max(1, N)


def accept(cur: Individual, cand: Individual, temp: float, cfg: Dict[str, Any]) -> bool:
    if better(cand, cur):
        return True
    delta = acceptance_delta(cand, cur, cfg)
    if delta <= 0:
        return True
    return rng.random() < math.exp(-delta / max(temp, cfg["min_temp"]))


# ---------------- main SA-guided improvement loop ----------------
def solve() -> Individual:
    cfg = get_config(N, K)
    start = time.perf_counter()
    end_time = start + cfg["time_limit"]
    main_end = start + cfg["time_limit"] * cfg["main_time_ratio"]
    pc = time.perf_counter

    cur = greedy_initial()
    cur = greedy_polish(cur, cfg, main_end, cfg["polish_passes"])
    best = clone(cur)

    temp = max(1.0, best.maxCost * cfg["t0_ratio"])
    no_improve = 0

    while pc() < main_end:
        cand = propose_candidate(cur, cfg, no_improve)
        if cand is None:
            temp = max(cfg["min_temp"], temp * cfg["cooling"])
            continue

        if better(cand, best):
            best = clone(cand)
            no_improve = 0
        else:
            no_improve += 1

        if accept(cur, cand, temp, cfg):
            cur = cand

        if no_improve >= cfg["restart_no_improve"] and pc() < main_end:
            shaken = candidate_ruin_recreate(best, cfg, no_improve)
            if shaken is not None:
                cur = greedy_polish(shaken, cfg, main_end, max(2, cfg["polish_passes"] // 5))
                if better(cur, best):
                    best = clone(cur)
            no_improve = 0
            temp = max(temp, best.maxCost * cfg["t0_ratio"] * 0.5)

        temp = max(cfg["min_temp"], temp * cfg["cooling"])

    if pc() < end_time:
        best = greedy_polish(best, cfg, end_time, cfg["final_polish_passes"])

    return make_individual(best.routes)


# ---------------- output ----------------
def print_solution(routes: List[List[int]]) -> None:
    print(K)
    for route in repair_routes(routes):
        print(len(route) - 1)
        print(*route[:-1])


def main() -> None:
    global N, K, SIZE, d, D
    N, K, d, D = input_data()
    if N <= 0 or K <= 0:
        return

    SIZE = N + 1

    solution = solve()
    print_solution(solution.routes)


if __name__ == "__main__":
    main()

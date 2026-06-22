import sys
import time
import random
import heapq
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any

INF = 10 ** 18
N = 0
K = 0
SIZE = 0
D: List[int] = []          
d: List[List[int]] = [] 
rng = random.Random(time.time_ns())

config = {
    "time_limit": 28.0,
    "update_period": 25,
    "weight_rho": 0.25,
}


@dataclass
class Individual:
    routes: List[List[int]] = field(default_factory=list)
    maxCost: int = 0
    totalCost: int = 0
    fitness: float = 0.0


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
    # Pure ALNS-style parameters. No SA temperature, no restart, no local-search phase.
    if n <= 250:
        return {
            "mode": "small",
            "time_limit": config["time_limit"],
            "update_period": config["update_period"],
            "weight_rho": config["weight_rho"],
            "destroy_min": max(2, n // 80),
            "destroy_max": max(8, n // 10),
            "repair_route_limit": k,
            "regret_limit": 40,
            "main_time_ratio": 0.995,
        }

    if n <= 600:
        return {
            "mode": "medium",
            "time_limit": config["time_limit"],
            "update_period": config["update_period"],
            "weight_rho": config["weight_rho"],
            "destroy_min": max(3, n // 100),
            "destroy_max": max(12, n // 14),
            "repair_route_limit": min(7, k),
            "regret_limit": 24,
            "main_time_ratio": 0.995,
        }

    return {
        "mode": "large",
        "time_limit": config["time_limit"],
        "update_period": config["update_period"],
        "weight_rho": 0.20,
        "destroy_min": max(3, n // 250),
        "destroy_max": max(12, n // 60),
        "repair_route_limit": min(5, k),
        "regret_limit": 0,
        "main_time_ratio": 0.995,
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
    return Individual([r[:] for r in ind.routes], ind.maxCost, ind.totalCost, ind.fitness)


def evaluate_clean(routes: List[List[int]]) -> Individual:
    # Use only when routes are already feasible and closed.
    costs = get_costs(routes)
    mx = max(costs) if costs else 0
    total = sum(costs)
    return Individual([r[:] for r in routes], mx, total, 1.0 / (mx + 1e-9))


def best_insertion(routes: List[List[int]], costs: List[int], x: int,
                   route_limit: int = None) -> Tuple[int, int, int, int, int]:
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


def best_two_insertions(routes: List[List[int]], costs: List[int], x: int) -> Tuple[Tuple[int, int, int, int, int], Tuple[int, int, int, int, int]]:
    flat = D
    s = SIZE
    old_total = sum(costs)

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

    best1 = (INF, INF, 0, 0, 1)
    best2 = (INF, INF, 0, 0, 1)

    for k in range(K):
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
            cand = (other_max if other_max >= new_cost else new_cost,
                    old_total + delta, delta, k, pos)
            if cand[:2] < best1[:2]:
                best2 = best1
                best1 = cand
            elif cand[:2] < best2[:2]:
                best2 = cand

    return best1, best2


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


# ---------------- destroy operators ----------------
def collect_positions(routes: List[List[int]]) -> List[Tuple[int, int, int]]:
    positions: List[Tuple[int, int, int]] = []
    for k, route in enumerate(routes):
        for i in range(1, len(route) - 1):
            positions.append((k, i, route[i]))
    return positions


def remove_selected(routes: List[List[int]], selected: List[Tuple[int, int, int]]) -> List[int]:
    removed: List[int] = []
    by_route: List[List[Tuple[int, int]]] = [[] for _ in range(K)]

    for k, i, x in selected:
        if 0 <= k < K:
            by_route[k].append((i, x))

    for k in range(K):
        if not by_route[k]:
            continue
        by_route[k].sort(reverse=True)
        route = routes[k]
        for i, x in by_route[k]:
            if 0 < i < len(route) - 1 and route[i] == x:
                route.pop(i)
                removed.append(x)
            else:
                try:
                    idx = route.index(x)
                    if 0 < idx < len(route) - 1:
                        route.pop(idx)
                        removed.append(x)
                except ValueError:
                    pass

    return removed


def destroy_random(routes: List[List[int]], q: int) -> List[int]:
    positions = collect_positions(routes)
    if not positions:
        return []
    rng.shuffle(positions)
    return remove_selected(routes, positions[:q])


def destroy_longest_route(routes: List[List[int]], q: int) -> List[int]:
    costs = get_costs(routes)
    order = list(range(K))
    order.sort(key=costs.__getitem__, reverse=True)

    flat = D
    s = SIZE
    selected: List[Tuple[int, int, int]] = []
    need = q

    for k in order:
        route = routes[k]
        if len(route) <= 2:
            continue
        candidates = []
        for i in range(1, len(route) - 1):
            p = route[i - 1]
            x = route[i]
            nxt = route[i + 1]
            saving = flat[p * s + x] + flat[x * s + nxt] - flat[p * s + nxt]
            candidates.append((saving, i, x))

        take = candidates if len(candidates) <= need else heapq.nlargest(need, candidates)
        for _saving, i, x in take:
            selected.append((k, i, x))
            if len(selected) >= q:
                break
        if len(selected) >= q:
            break
        need = q - len(selected)

    return remove_selected(routes, selected)


def destroy_longest_segment(routes: List[List[int]], q: int) -> List[int]:
    costs = get_costs(routes)
    longest = max(range(K), key=costs.__getitem__)
    route = routes[longest]
    customer_count = len(route) - 2
    if customer_count <= 0:
        return []

    q = min(q, customer_count)
    start = rng.randint(1, len(route) - 1 - q)
    selected = [(longest, i, route[i]) for i in range(start, start + q)]
    return remove_selected(routes, selected)


def destroy_worst(routes: List[List[int]], q: int) -> List[int]:
    flat = D
    s = SIZE
    candidates: List[Tuple[int, int, int, int]] = []

    for k, route in enumerate(routes):
        for i in range(1, len(route) - 1):
            p = route[i - 1]
            x = route[i]
            nxt = route[i + 1]
            saving = flat[p * s + x] + flat[x * s + nxt] - flat[p * s + nxt]
            candidates.append((saving, k, i, x))

    if len(candidates) > q:
        candidates = heapq.nlargest(q, candidates)
    selected = [(k, i, x) for _saving, k, i, x in candidates]
    return remove_selected(routes, selected)


def destroy_related(routes: List[List[int]], q: int) -> List[int]:
    positions = collect_positions(routes)
    if not positions:
        return []

    flat = D
    s = SIZE
    _, _, seed = rng.choice(positions)
    if len(positions) > q:
        positions = heapq.nsmallest(q, positions, key=lambda item: flat[seed * s + item[2]])
    else:
        positions.sort(key=lambda item: flat[seed * s + item[2]])
    return remove_selected(routes, positions)


# ---------------- repair operators ----------------
def repair_minmax(routes: List[List[int]], removed: List[int], route_limit: int = None) -> None:
    costs = get_costs(routes)
    nodes = removed[:]
    nodes.sort(key=lambda x: D[x], reverse=True)

    for x in nodes:
        _, _, delta, k, pos = best_insertion(routes, costs, x, route_limit)
        routes[k].insert(pos, x)
        costs[k] += delta


def repair_regret2(routes: List[List[int]], removed: List[int], end_time: float) -> None:
    costs = get_costs(routes)
    remaining = removed[:]

    while remaining and time.perf_counter() < end_time:
        best_idx = 0
        best_choice = None
        best_regret = (-1, -1)

        for idx, x in enumerate(remaining):
            opt1, opt2 = best_two_insertions(routes, costs, x)
            regret = (opt2[0] - opt1[0], opt2[1] - opt1[1])
            if regret > best_regret:
                best_regret = regret
                best_idx = idx
                best_choice = opt1

        x = remaining.pop(best_idx)
        if best_choice is None:
            best_choice = best_insertion(routes, costs, x, K)

        _, _, delta, k, pos = best_choice
        routes[k].insert(pos, x)
        costs[k] += delta

    if remaining:
        repair_minmax(routes, remaining, min(5, K))


# ---------------- adaptive operator choice ----------------
def choose_operator(weights: List[float]) -> int:
    total = sum(weights)
    if total <= 0:
        return rng.randrange(len(weights))

    pick = rng.random() * total
    cur = 0.0
    for i, w in enumerate(weights):
        cur += w
        if cur >= pick:
            return i
    return len(weights) - 1


def update_weights(weights: List[float], scores: List[float], counts: List[int], rho: float) -> None:
    for i in range(len(weights)):
        if counts[i] > 0:
            avg = scores[i] / counts[i]
            weights[i] = max(0.1, (1.0 - rho) * weights[i] + rho * avg)
        scores[i] = 0.0
        counts[i] = 0


# ---------------- ALNS acceptance ----------------
def accept(cur: Individual, cand: Individual) -> bool:
    return better(cand, cur)


def apply_destroy(op_id: int, routes: List[List[int]], q: int) -> List[int]:
    if op_id == 0:
        return destroy_longest_route(routes, q)
    if op_id == 1:
        return destroy_longest_segment(routes, q)
    if op_id == 2:
        return destroy_worst(routes, q)
    if op_id == 3:
        return destroy_random(routes, q)
    return destroy_related(routes, q)


def apply_repair(op_id: int, routes: List[List[int]], removed: List[int],
                 cfg: Dict[str, Any], end_time: float) -> None:
    if op_id == 0:
        repair_minmax(routes, removed, cfg["repair_route_limit"])
    elif op_id == 1:
        repair_minmax(routes, removed, K)
    else:
        if cfg["regret_limit"] > 0 and len(removed) <= cfg["regret_limit"]:
            repair_regret2(routes, removed, end_time)
        else:
            repair_minmax(routes, removed, cfg["repair_route_limit"])


# ---------------- main ALNS loop ----------------
def solve() -> Individual:
    cfg = get_config(N, K)
    start = time.perf_counter()
    end_time = start + cfg["time_limit"]
    main_end = start + cfg["time_limit"] * cfg["main_time_ratio"]

    cur = greedy_initial()
    best = clone(cur)

    destroy_ids = [0, 1, 2, 3, 4]
    repair_ids = [0, 1] if cfg["mode"] == "large" else [1, 2, 0]

    destroy_weights = [1.0] * len(destroy_ids)
    repair_weights = [1.0] * len(repair_ids)
    destroy_scores = [0.0] * len(destroy_ids)
    repair_scores = [0.0] * len(repair_ids)
    destroy_counts = [0] * len(destroy_ids)
    repair_counts = [0] * len(repair_ids)

    no_improve = 0
    it = 0

    while time.perf_counter() < main_end:
        it += 1
        routes = [r[:] for r in cur.routes]

        q_low = max(1, min(cfg["destroy_min"], N))
        q_high = max(q_low, min(cfg["destroy_max"], N))
        if no_improve > cfg["update_period"]:
            q_high = min(N, int(q_high * 1.5) + 1)

        q = rng.randint(q_low, q_high)
        d_idx = choose_operator(destroy_weights)
        r_idx = choose_operator(repair_weights)

        removed = apply_destroy(destroy_ids[d_idx], routes, q)
        if not removed:
            continue

        apply_repair(repair_ids[r_idx], routes, removed, cfg, main_end)
        cand = evaluate_clean(routes)

        reward = 0.0
        if better(cand, best):
            best = clone(cand)
            reward = 8.0
            no_improve = 0
        elif better(cand, cur):
            reward = 4.0
            no_improve += 1
        else:
            no_improve += 1

        if accept(cur, cand):
            cur = cand
            if reward == 0.0:
                reward = 1.0

        destroy_scores[d_idx] += reward
        repair_scores[r_idx] += reward
        destroy_counts[d_idx] += 1
        repair_counts[r_idx] += 1

        if it % cfg["update_period"] == 0:
            update_weights(destroy_weights, destroy_scores, destroy_counts, cfg["weight_rho"])
            update_weights(repair_weights, repair_scores, repair_counts, cfg["weight_rho"])

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

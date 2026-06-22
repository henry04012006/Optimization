import sys
import time
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional

INF = 1 << 60
N = 0
K = 0
SIZE = 0
D: List[int] = []                  
d: List[List[int]] = []            
rng = random.Random(time.time_ns())


@dataclass
class Individual:
    routes: List[List[int]] = field(default_factory=list)
    maxCost: int = 0
    totalCost: int = 0
    fitness: float = 0.0


#----------------get input/config----------------
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
        dist.append(row)
        flat.extend(row)

    return n, k, dist, flat


def get_config(n: int, k: int) -> Dict[str, Any]:
    # pheromone initialization, ant construction, pheromone update, and time-limit stopping.
    if n <= 250:
        return dict(
            mode="small",
            time_limit=28.75,
            reserve_time=0.15,
            num_ants=36,
            elite_ants=6,
            alpha=1.0,
            beta=3.0,
            evaporation=0.85,
            deposit=28.0,
            best_deposit=2.5,
            tau0=1.0,
            tau_min=0.03,
            tau_max=35.0,
            init_route_limit=k,
            route_limit=k,
        )

    if n <= 600:
        return dict(
            mode="medium",
            time_limit=28.75,
            reserve_time=0.15,
            num_ants=24,
            elite_ants=5,
            alpha=1.0,
            beta=2.8,
            evaporation=0.87,
            deposit=24.0,
            best_deposit=2.2,
            tau0=1.0,
            tau_min=0.03,
            tau_max=30.0,
            init_route_limit=min(8, k),
            route_limit=min(8, k),
        )

    return dict(
        mode="large",
        time_limit=28.75,
        reserve_time=0.20,
        num_ants=14,
        elite_ants=3,
        alpha=1.0,
        beta=2.6,
        evaporation=0.89,
        deposit=18.0,
        best_deposit=2.0,
        tau0=1.0,
        tau_min=0.03,
        tau_max=25.0,
        init_route_limit=min(4, k),
        route_limit=min(5, k),
    )


#----------------helper functions: basic evaluation----------------
def route_cost(route: List[int]) -> int:
    dd = D
    size = SIZE
    cost = 0

    for i in range(len(route) - 1):
        cost += dd[route[i] * size + route[i + 1]]

    return cost


def get_costs(routes: List[List[int]]) -> List[int]:
    return [route_cost(route) for route in routes]


def evaluate_clean(routes: List[List[int]], costs: Optional[List[int]] = None) -> Individual:
    if costs is None:
        costs = get_costs(routes)

    mx = max(costs) if costs else 0
    total = sum(costs)

    return Individual([r[:] for r in routes], mx, total, 1.0 / (mx + 1e-9))


def clone(ind: Individual) -> Individual:
    return Individual([r[:] for r in ind.routes], ind.maxCost, ind.totalCost, ind.fitness)


def better(a: Individual, b: Individual) -> bool:
    return (a.maxCost, a.totalCost) < (b.maxCost, b.totalCost)


#----------------helper functions: insertion/feasibility repair----------------
def best_insertion(routes: List[List[int]], costs: List[int], x: int,
                   route_limit: Optional[int] = None) -> Tuple[int, int, int, int, int]:
    dd = D
    size = SIZE
    total = sum(costs)

    max1 = -1
    max2 = -1
    max_idx = -1

    for i, c in enumerate(costs):
        if c > max1:
            max2 = max1
            max1 = c
            max_idx = i
        elif c > max2:
            max2 = c

    if route_limit is not None and route_limit < K:
        route_order = sorted(range(K), key=costs.__getitem__)[:route_limit]
    else:
        route_order = range(K)

    best_new_max = INF
    best_new_total = INF
    best_delta = 0
    best_k = 0
    best_pos = 1

    for k in route_order:
        other_max = max2 if k == max_idx else max1
        route = routes[k]
        old_cost = costs[k]

        for pos in range(1, len(route)):
            a = route[pos - 1]
            b = route[pos]

            delta = dd[a * size + x] + dd[x * size + b] - dd[a * size + b]
            new_cost = old_cost + delta
            new_max = other_max if other_max > new_cost else new_cost
            new_total = total + delta

            if new_max < best_new_max or (new_max == best_new_max and new_total < best_new_total):
                best_new_max = new_max
                best_new_total = new_total
                best_delta = delta
                best_k = k
                best_pos = pos

    return best_new_max, best_new_total, best_delta, best_k, best_pos


def repair_routes(routes: List[List[int]]) -> List[List[int]]:
    if K <= 0:
        return []

    routes = [r[:] for r in routes[:K]]
    while len(routes) < K:
        routes.append([0, 0])

    clean: List[List[int]] = []
    seen = [False] * (N + 1)

    for route in routes:
        new_route = [0]

        for x in route:
            if 1 <= x <= N and not seen[x]:
                new_route.append(x)
                seen[x] = True

        new_route.append(0)
        clean.append(new_route)

    missing = [x for x in range(1, N + 1) if not seen[x]]

    if missing:
        costs = get_costs(clean)
        missing.sort(key=lambda x: D[x], reverse=True)

        for x in missing:
            _, _, delta, k, pos = best_insertion(clean, costs, x, K)
            clean[k].insert(pos, x)
            costs[k] += delta

    return clean


def make_individual(routes: List[List[int]]) -> Individual:
    return evaluate_clean(repair_routes(routes))


# ----------------initial solution----------------
def greedy_initial(route_limit: Optional[int] = None) -> Individual:
    routes = [[0, 0] for _ in range(K)]
    costs = [0] * K
    nodes = list(range(1, N + 1))

    # Insert far pickup points first because they usually affect maxCost more.
    nodes.sort(key=lambda x: D[x], reverse=True)

    for x in nodes:
        _, _, delta, k, pos = best_insertion(routes, costs, x, route_limit)
        routes[k].insert(pos, x)
        costs[k] += delta

    return evaluate_clean(routes, costs)


# --------------- pheromone functions----------------
def edge_key(u: int, v: int) -> int:
    return u * SIZE + v


def get_tau(pheromone: Dict[int, float], base_tau: float, u: int, v: int) -> float:
    return pheromone.get(edge_key(u, v), base_tau)


def add_tau(pheromone: Dict[int, float], base_tau: float,
            u: int, v: int, amount: float, cfg: Dict[str, Any]) -> None:
    key = edge_key(u, v)
    value = pheromone.get(key, base_tau) + amount

    if value > cfg["tau_max"]:
        value = cfg["tau_max"]

    pheromone[key] = value


def evaporate_pheromone(pheromone: Dict[int, float], base_tau: float,
                        cfg: Dict[str, Any]) -> float:
    evaporation = cfg["evaporation"]
    tau_min = cfg["tau_min"]

    new_base = base_tau * evaporation
    if new_base < tau_min:
        new_base = tau_min

    remove_keys: List[int] = []

    for key, value in pheromone.items():
        new_value = value * evaporation
        if new_value < tau_min:
            new_value = tau_min

        if new_value <= new_base * 1.000001:
            remove_keys.append(key)
        else:
            pheromone[key] = new_value

    for key in remove_keys:
        del pheromone[key]

    return new_base


def deposit_solution(pheromone: Dict[int, float], base_tau: float,
                     ind: Individual, amount: float, cfg: Dict[str, Any]) -> None:
    if amount <= 0.0:
        return

    for route in ind.routes:
        for i in range(len(route) - 1):
            u = route[i]
            v = route[i + 1]

            if u == 0 and v == 0:
                continue

            add_tau(pheromone, base_tau, u, v, amount, cfg)
            add_tau(pheromone, base_tau, v, u, amount, cfg)


def update_pheromone(pheromone: Dict[int, float], base_tau: float,
                     ants: List[Individual], best: Individual,
                     cfg: Dict[str, Any]) -> float:
    base_tau = evaporate_pheromone(pheromone, base_tau, cfg)

    if not ants:
        return base_tau

    ants.sort(key=lambda ind: (ind.maxCost, ind.totalCost))
    selected = ants[:min(cfg["elite_ants"], len(ants))]
    best_ref = max(1, best.maxCost)

    for ant in selected:
        quality = best_ref / max(1, ant.maxCost)
        amount = cfg["deposit"] * quality / max(1, N)
        deposit_solution(pheromone, base_tau, ant, amount, cfg)

    best_amount = cfg["deposit"] * cfg["best_deposit"] / max(1, N)
    deposit_solution(pheromone, base_tau, best, best_amount, cfg)

    return base_tau


#----------------ant construction----------------
def choose_next_customer(current: int, remaining: List[int],
                         pheromone: Dict[int, float], base_tau: float,
                         cfg: Dict[str, Any]) -> int:
    dd = D
    size = SIZE
    alpha = cfg["alpha"]
    beta = cfg["beta"]

    values: List[float] = []
    total = 0.0

    for x in remaining:
        distance = dd[current * size + x]
        eta = 1.0 / (distance + 1.0)
        tau = get_tau(pheromone, base_tau, current, x)
        value = (tau ** alpha) * (eta ** beta)

        values.append(value)
        total += value

    if total <= 0.0:
        return remaining[rng.randrange(len(remaining))]

    pick = rng.random() * total
    cur = 0.0

    for x, value in zip(remaining, values):
        cur += value
        if cur >= pick:
            return x

    return remaining[-1]


def construct_ant_solution(pheromone: Dict[int, float],
                           base_tau: float,
                           cfg: Dict[str, Any]) -> Individual:
    routes = [[0, 0] for _ in range(K)]
    costs = [0] * K
    remaining = list(range(1, N + 1))

    current = 0

    while remaining:
        x = choose_next_customer(current, remaining, pheromone, base_tau, cfg)
        remaining.remove(x)

        _, _, delta, k, pos = best_insertion(routes, costs, x, cfg["route_limit"])
        routes[k].insert(pos, x)
        costs[k] += delta

        current = x

    return evaluate_clean(routes, costs)


#----------------main ACO loop----------------
def solve() -> Individual:
    cfg = get_config(N, K)
    start = time.perf_counter()
    end_time = start + cfg["time_limit"] - cfg["reserve_time"]
    pc = time.perf_counter

    best = greedy_initial(cfg["init_route_limit"])

    pheromone: Dict[int, float] = {}
    base_tau = cfg["tau0"]

    while pc() < end_time:
        ants: List[Individual] = []

        for _ in range(cfg["num_ants"]):
            if pc() >= end_time:
                break

            ant = construct_ant_solution(pheromone, base_tau, cfg)
            ants.append(ant)

            if better(ant, best):
                best = clone(ant)

        if ants:
            base_tau = update_pheromone(pheromone, base_tau, ants, best, cfg)

    return make_individual(best.routes)


# ----------------output----------------
def print_solution(routes: List[List[int]]) -> None:
    print(K)

    for route in repair_routes(routes):
        # Internal route is closed: [0, pickup points..., 0]
        # Output skips only the final depot.
        print(len(route) - 1)
        print(*route[:-1])


def main() -> None:
    global N, K, SIZE, D, d

    N, K, d, D = input_data()
    if N <= 0 or K <= 0:
        return

    SIZE = N + 1

    solution = solve()
    print_solution(solution.routes)
    print("Objective:", solution.maxCost)


if __name__ == "__main__":
    main()

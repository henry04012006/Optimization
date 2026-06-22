import sys
import time
import random
import heapq
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
    "time_limit": 28.75,
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
    if n <= 100:
        return dict(
            mode="tiny", pop_size=70, elite=4, time_limit=config["time_limit"],
            init_mc=26, child_mc=12, best_mc=32, final_mc=45,
            memetic_rate=0.45, memetic_rounds=2, best_every=4,
            final_rounds=16, final_best_count=4,
            crossover_rate=0.90, mutation_rate=0.25,
            no_improve_limit=25, replace_div=7,
            relocate_sample=18, relocate_dest=min(7, max(1, k - 1)),
            route_limit=k, init_route_limit=k, mixed=True,
            tournament_size=3, reserve_time=0.12,
        )

    if n <= 300:
        return dict(
            mode="small", pop_size=60, elite=4, time_limit=config["time_limit"],
            init_mc=26, child_mc=10, best_mc=32, final_mc=36,
            memetic_rate=0.35, memetic_rounds=2, best_every=4,
            final_rounds=12, final_best_count=4,
            crossover_rate=0.90, mutation_rate=0.25,
            no_improve_limit=32, replace_div=7,
            relocate_sample=18, relocate_dest=min(7, max(1, k - 1)),
            route_limit=k, init_route_limit=k, mixed=True,
            tournament_size=3, reserve_time=0.12,
        )

    if n <= 600:
        return dict(
            mode="medium", pop_size=48, elite=3, time_limit=config["time_limit"],
            init_mc=16, child_mc=7, best_mc=20, final_mc=28,
            memetic_rate=0.22, memetic_rounds=1, best_every=5,
            final_rounds=8, final_best_count=3,
            crossover_rate=0.84, mutation_rate=0.25,
            no_improve_limit=26, replace_div=7,
            relocate_sample=13, relocate_dest=min(5, max(1, k - 1)),
            route_limit=min(8, k), init_route_limit=min(8, k), mixed=True,
            tournament_size=3, reserve_time=0.12,
        )

    if n <= 850:
        return dict(
            mode="medium_large", pop_size=36, elite=2, time_limit=config["time_limit"],
            init_mc=6, child_mc=4, best_mc=6, final_mc=14,
            memetic_rate=0.12, memetic_rounds=1, best_every=7,
            final_rounds=4, final_best_count=2,
            crossover_rate=0.74, mutation_rate=0.28,
            no_improve_limit=22, replace_div=6,
            relocate_sample=8, relocate_dest=min(4, max(1, k - 1)),
            route_limit=min(5, k), init_route_limit=min(4, k), mixed=False,
            tournament_size=3, reserve_time=0.15,
        )
    return dict(
        mode="large", pop_size=30, elite=2, time_limit=config["time_limit"],
        init_mc=6, child_mc=3, best_mc=6, final_mc=8,
        memetic_rate=0.07, memetic_rounds=1, best_every=7,
        final_rounds=2, final_best_count=2,
        crossover_rate=0.74, mutation_rate=0.28,
        no_improve_limit=22, replace_div=6,
        relocate_sample=8, relocate_dest=min(4, max(1, k - 1)),
        route_limit=min(5, k), init_route_limit=min(4, k), mixed=False,
        tournament_size=3, reserve_time=0.15,
    )


# ---------------- basic evaluation ----------------
def route_cost(route: List[int]) -> int:
    dd = D
    size = SIZE
    cost = 0
    for i in range(len(route) - 1):
        cost += dd[route[i] * size + route[i + 1]]
    return cost


def get_costs(routes: List[List[int]]) -> List[int]:
    return [route_cost(r) for r in routes]


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


def insert_delta(route: List[int], pos: int, x: int) -> int:
    dd = D
    size = SIZE
    a = route[pos - 1]
    b = route[pos]
    return dd[a * size + x] + dd[x * size + b] - dd[a * size + b]


def remove_delta(route: List[int], idx: int) -> int:
    dd = D
    size = SIZE
    p = route[idx - 1]
    x = route[idx]
    q = route[idx + 1]
    return dd[p * size + q] - dd[p * size + x] - dd[x * size + q]


# ---------------- repair / insertion ----------------
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
        order = sorted(range(K), key=costs.__getitem__)[:route_limit]
    else:
        order = range(K)

    best_new_max = INF
    best_new_total = INF
    best_delta = 0
    best_k = 0
    best_pos = 1

    for k in order:
        other_max = max2 if k == max_idx else max1
        r = routes[k]
        ck = costs[k]
        for pos in range(1, len(r)):
            a = r[pos - 1]
            b = r[pos]
            delta = dd[a * size + x] + dd[x * size + b] - dd[a * size + b]
            new_cost = ck + delta
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

    for r in routes:
        nr = [0]
        for x in r:
            if 1 <= x <= N and not seen[x]:
                nr.append(x)
                seen[x] = True
        nr.append(0)
        clean.append(nr)

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
    routes = repair_routes(routes)
    return evaluate_clean(routes)


# ---------------- initialization ----------------
def balanced_greedy_ini(mode: int, route_limit: Optional[int] = None, noise_rate: float = 0.0) -> List[List[int]]:
    routes = [[0, 0] for _ in range(K)]
    costs = [0] * K
    nodes = list(range(1, N + 1))

    if mode == 1:
        nodes.sort(key=lambda x: D[x], reverse=True)
    elif mode == 2:
        nodes.sort(key=lambda x: D[x])
    else:
        rng.shuffle(nodes)

    if noise_rate > 0.0 and len(nodes) > 3:
        block = max(4, int(len(nodes) * noise_rate))
        for start in range(0, len(nodes), block):
            part = nodes[start:start + block]
            rng.shuffle(part)
            nodes[start:start + block] = part

    if route_limit is None:
        route_limit = K

    for x in nodes:
        _, _, delta, k, pos = best_insertion(routes, costs, x, route_limit)
        routes[k].insert(pos, x)
        costs[k] += delta

    return routes


def perturb_routes(routes: List[List[int]], moves: int) -> List[List[int]]:
    routes = [r[:] for r in routes]
    for _ in range(moves):
        non_empty = [k for k in range(K) if len(routes[k]) > 2]
        if not non_empty:
            break
        a = rng.choice(non_empty)
        i = rng.randint(1, len(routes[a]) - 2)
        x = routes[a].pop(i)
        b = rng.randrange(K)
        pos = rng.randint(1, len(routes[b]) - 1)
        routes[b].insert(pos, x)
    return routes


def initial_population(size: int, cfg: Dict[str, Any], end_time: float) -> List[Individual]:
    pop: List[Individual] = []
    pc = time.perf_counter

    for i in range(size):
        if pc() >= end_time:
            break

        mode = i % 3
        noise = 0.0
        if i >= 3:
            noise = 0.015 * (1 + (i % 4))

        routes = balanced_greedy_ini(mode, cfg["init_route_limit"], noise)
        if i >= 3:
            moves = 1 + (i % max(2, N // 120 + 1))
            routes = perturb_routes(routes, moves)

        mc = cfg["init_mc"] if i < 3 else max(2, cfg["init_mc"] // 2)
        routes = two_opt_all(routes, mc, end_time)
        pop.append(evaluate_clean(routes))

    if not pop:
        routes = balanced_greedy_ini(1, cfg["init_route_limit"])
        pop.append(evaluate_clean(routes))

    pop.sort(key=lambda ind: (ind.maxCost, ind.totalCost))
    return pop[:size]


# ---------------- local search ----------------
def two_opt_route(route: List[int], max_attempt: int, end_time: Optional[float] = None) -> List[int]:
    if max_attempt <= 0 or len(route) < 4:
        return route[:]

    dd = D
    size = SIZE
    cur = route[:]
    n = len(cur)
    t = 0
    pc = time.perf_counter

    while t < max_attempt:
        if end_time is not None and pc() >= end_time:
            break
        i = rng.randint(0, n - 4)
        j = rng.randint(i + 2, n - 2)
        a = cur[i]
        b = cur[i + 1]
        c = cur[j]
        e = cur[j + 1]
        delta = dd[a * size + c] + dd[b * size + e] - dd[a * size + b] - dd[c * size + e]
        if delta < 0:
            cur[i + 1:j + 1] = reversed(cur[i + 1:j + 1])
            t = 0
        else:
            t += 1

    return cur


def two_opt_all(routes: List[List[int]], max_attempt: int, end_time: Optional[float] = None) -> List[List[int]]:
    if max_attempt <= 0:
        return [r[:] for r in routes]
    pc = time.perf_counter
    ans: List[List[int]] = []
    for r in routes:
        if end_time is not None and pc() >= end_time:
            ans.append(r[:])
        else:
            ans.append(two_opt_route(r, max_attempt, end_time))
    return ans


def relocate_from_longest(routes: List[List[int]], costs: List[int],
                          sample_nodes: int, max_dest_routes: int,
                          mixed: bool) -> bool:
    dd = D
    size = SIZE
    longest = max(range(K), key=costs.__getitem__)
    if len(routes[longest]) <= 3:
        return False

    old_max = max(costs)
    old_total = sum(costs)
    r_long = routes[longest]

    candidates: List[Tuple[int, int]] = []
    for i in range(1, len(r_long) - 1):
        p = r_long[i - 1]
        x = r_long[i]
        q = r_long[i + 1]
        rem = dd[p * size + q] - dd[p * size + x] - dd[x * size + q]
        candidates.append((rem, i))

    if len(candidates) > sample_nodes:
        if mixed:
            keep = max(1, sample_nodes // 2)
            candidates.sort()
            rest = candidates[keep:]
            rng.shuffle(rest)
            candidates = candidates[:keep] + rest[:sample_nodes - keep]
        else:
            candidates = heapq.nsmallest(sample_nodes, candidates)
    else:
        candidates.sort()

    dest = [k for k in range(K) if k != longest]
    dest.sort(key=costs.__getitem__)
    dest = dest[:min(max_dest_routes, len(dest))]

    best_info = None
    best_max = old_max
    best_total = old_total

    for _rem0, i in candidates:
        x = r_long[i]
        p = r_long[i - 1]
        q = r_long[i + 1]
        rem = dd[p * size + q] - dd[p * size + x] - dd[x * size + q]
        new_long = costs[longest] + rem

        for k in dest:
            other_max = 0
            for t in range(K):
                if t != longest and t != k and costs[t] > other_max:
                    other_max = costs[t]
            r = routes[k]
            ck = costs[k]
            for pos in range(1, len(r)):
                a = r[pos - 1]
                b = r[pos]
                ins = dd[a * size + x] + dd[x * size + b] - dd[a * size + b]
                new_k = ck + ins
                new_max = max(new_long, new_k, other_max)
                new_total = old_total + rem + ins
                if new_max < best_max or (new_max == best_max and new_total < best_total):
                    best_max = new_max
                    best_total = new_total
                    best_info = (i, k, pos, rem, ins)

    if best_info is None:
        return False

    i, k, pos, rem, ins = best_info
    x = routes[longest].pop(i)
    routes[k].insert(pos, x)
    costs[longest] += rem
    costs[k] += ins
    return True


def improve(ind: Individual, cfg: Dict[str, Any], rounds: int,
            mc_attempt: Optional[int] = None, end_time: Optional[float] = None) -> Individual:
    pc = time.perf_counter
    best = clone(ind)
    mc = cfg["child_mc"] if mc_attempt is None else mc_attempt

    for _ in range(rounds):
        if end_time is not None and pc() >= end_time:
            break
        routes = [r[:] for r in best.routes]
        costs = get_costs(routes)
        changed = relocate_from_longest(routes, costs, cfg["relocate_sample"], cfg["relocate_dest"], cfg["mixed"])
        if not changed:
            break
        routes = two_opt_all(routes, mc, end_time)
        cand = evaluate_clean(routes)
        if better(cand, best):
            best = cand
        else:
            break

    return best


def random_relocate_inplace(routes: List[List[int]], moves: int = 1) -> None:
    for _ in range(moves):
        non_empty = [k for k in range(K) if len(routes[k]) > 2]
        if not non_empty:
            return
        a = rng.choice(non_empty)
        i = rng.randint(1, len(routes[a]) - 2)
        x = routes[a].pop(i)
        b = rng.randrange(K)
        pos = rng.randint(1, len(routes[b]) - 1)
        routes[b].insert(pos, x)


def random_swap_inplace(routes: List[List[int]]) -> None:
    non_empty = [k for k in range(K) if len(routes[k]) > 2]
    if not non_empty:
        return
    a = rng.choice(non_empty)
    b = rng.choice(non_empty)
    ia = rng.randint(1, len(routes[a]) - 2)
    ib = rng.randint(1, len(routes[b]) - 2)
    routes[a][ia], routes[b][ib] = routes[b][ib], routes[a][ia]


def route_reverse_mutation(routes: List[List[int]]) -> None:
    non_empty = [k for k in range(K) if len(routes[k]) > 4]
    if not non_empty:
        return
    k = rng.choice(non_empty)
    r = routes[k]
    i = rng.randint(1, len(r) - 3)
    j = rng.randint(i + 1, len(r) - 2)
    r[i:j + 1] = reversed(r[i:j + 1])


# ---------------- crossover / mutation ----------------
def flatten(routes: List[List[int]]) -> List[int]:
    return [x for r in routes for x in r[1:-1]]


def route_preserving_child(a: Individual, b: Individual, cfg: Dict[str, Any]) -> Individual:
    child = [[0, 0] for _ in range(K)]
    used = [False] * (N + 1)

    idxs = list(range(K))
    if rng.random() < 0.65:
        ac = get_costs(a.routes)
        idxs.sort(key=ac.__getitem__)
    else:
        rng.shuffle(idxs)

    copy_max = max(1, K // 3)
    copy_cnt = rng.randint(1, copy_max) if K > 1 else 1

    for k in idxs[:copy_cnt]:
        dst = child[k]
        for x in a.routes[k][1:-1]:
            if not used[x]:
                dst.insert(len(dst) - 1, x)
                used[x] = True

    costs = get_costs(child)
    for x in flatten(b.routes):
        if not used[x]:
            _, _, delta, k, pos = best_insertion(child, costs, x, cfg["route_limit"])
            child[k].insert(pos, x)
            costs[k] += delta
            used[x] = True

    return evaluate_clean(child, costs)


def order_child(a: Individual, b: Individual, cfg: Dict[str, Any]) -> Individual:
    order_a = flatten(a.routes)
    order_b = flatten(b.routes)
    if N <= 2:
        return route_preserving_child(a, b, cfg)

    l = rng.randint(0, N - 1)
    r = rng.randint(l, N - 1)
    used = [False] * (N + 1)
    order: List[int] = []

    for x in order_a[l:r + 1]:
        if not used[x]:
            order.append(x)
            used[x] = True
    for x in order_b:
        if not used[x]:
            order.append(x)
            used[x] = True

    routes = [[0, 0] for _ in range(K)]
    costs = [0] * K
    for x in order:
        _, _, delta, k, pos = best_insertion(routes, costs, x, cfg["route_limit"])
        routes[k].insert(pos, x)
        costs[k] += delta

    return evaluate_clean(routes, costs)


def crossover(p1: Individual, p2: Individual, cfg: Dict[str, Any]) -> Tuple[Individual, Individual]:
    if rng.random() < 0.82:
        return route_preserving_child(p1, p2, cfg), route_preserving_child(p2, p1, cfg)
    return order_child(p1, p2, cfg), order_child(p2, p1, cfg)


def mutate(ind: Individual, cfg: Dict[str, Any], end_time: Optional[float] = None) -> Individual:
    routes = [r[:] for r in ind.routes]
    p = rng.random()

    if p < 0.45:
        costs = get_costs(routes)
        relocate_from_longest(routes, costs, cfg["relocate_sample"], cfg["relocate_dest"], cfg["mixed"])
    elif p < 0.70:
        random_relocate_inplace(routes, 1)
    elif p < 0.88:
        random_swap_inplace(routes)
    else:
        route_reverse_mutation(routes)

    if rng.random() < 0.18:
        routes = two_opt_all(routes, max(1, cfg["child_mc"] // 2), end_time)

    return evaluate_clean(routes)


# ---------------- selection / population ----------------
def select(pop: List[Individual], tsize: int) -> Individual:
    best = rng.choice(pop)
    for _ in range(tsize - 1):
        cand = rng.choice(pop)
        if better(cand, best):
            best = cand
    return best


def inject_diversity(pop: List[Individual], cfg: Dict[str, Any], start_index: int, end_time: float) -> None:
    pc = time.perf_counter
    for i in range(start_index, len(pop)):
        if pc() >= end_time:
            break
        mode = i % 3
        routes = balanced_greedy_ini(mode, cfg["init_route_limit"], noise_rate=0.02 * (1 + i % 4))
        random_relocate_inplace(routes, moves=max(1, N // 180))
        routes = two_opt_all(routes, max(2, cfg["init_mc"] // 2), end_time)
        pop[i] = evaluate_clean(routes)


# ---------------- main memetic algorithm ----------------
def solve() -> Individual:
    cfg = get_config(N, K)
    start = time.perf_counter()
    end_time = start + cfg["time_limit"] - cfg["reserve_time"]

    pop = initial_population(cfg["pop_size"], cfg, end_time)
    best = clone(pop[0])
    no_improve = 0
    gen = 0
    pc = time.perf_counter

    while pc() < end_time:
        gen += 1
        pop.sort(key=lambda ind: (ind.maxCost, ind.totalCost))
        new_pop = [clone(ind) for ind in pop[:cfg["elite"]]]

        while len(new_pop) < cfg["pop_size"] and pc() < end_time:
            p1 = select(pop, cfg["tournament_size"])
            p2 = select(pop, cfg["tournament_size"])

            if rng.random() < cfg["crossover_rate"]:
                c1, c2 = crossover(p1, p2, cfg)
            else:
                c1, c2 = clone(p1), clone(p2)

            if rng.random() < cfg["mutation_rate"]:
                c1 = mutate(c1, cfg, end_time)
            if rng.random() < cfg["mutation_rate"] and len(new_pop) + 1 < cfg["pop_size"]:
                c2 = mutate(c2, cfg, end_time)

            if rng.random() < cfg["memetic_rate"]:
                c1 = improve(c1, cfg, cfg["memetic_rounds"], end_time=end_time)
            if rng.random() < cfg["memetic_rate"] and len(new_pop) + 1 < cfg["pop_size"]:
                c2 = improve(c2, cfg, cfg["memetic_rounds"], end_time=end_time)

            new_pop.append(c1)
            if len(new_pop) < cfg["pop_size"]:
                new_pop.append(c2)

        if not new_pop:
            break
        pop = new_pop
        cur = min(pop, key=lambda ind: (ind.maxCost, ind.totalCost))

        if gen % cfg["best_every"] == 0 and pc() < end_time:
            cur = improve(cur, cfg, 1, cfg["best_mc"], end_time)
            worst = max(range(len(pop)), key=lambda i: (pop[i].maxCost, pop[i].totalCost))
            pop[worst] = cur

        if better(cur, best):
            best = clone(cur)
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= cfg["no_improve_limit"] and pc() < end_time:
            pop.sort(key=lambda ind: (ind.maxCost, ind.totalCost))
            keep = max(cfg["elite"], cfg["pop_size"] - max(2, cfg["pop_size"] // cfg["replace_div"]))
            inject_diversity(pop, cfg, keep, end_time)
            no_improve = 0

    pop.sort(key=lambda ind: (ind.maxCost, ind.totalCost))
    for cand in pop[:min(cfg["final_best_count"], len(pop))]:
        if pc() >= end_time:
            break
        cand = improve(cand, cfg, cfg["final_rounds"], cfg["final_mc"], end_time)
        if better(cand, best):
            best = cand

    return make_individual(best.routes)


# ---------------- output ----------------
def print_solution(routes: List[List[int]]) -> None:
    print(K)
    for r in repair_routes(routes):
        print(len(r) - 1)
        print(*r[:-1])


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

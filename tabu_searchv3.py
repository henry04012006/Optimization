import random
import sys
import time


TIME_LIMIT_SECONDS = 29.0
RANDOM_SEED = 1
INF = 10**18


def read_input():
    numbers = list(map(int, sys.stdin.buffer.read().split()))
    position = 0
    number_of_points = numbers[position]
    position += 1
    number_of_postmen = numbers[position]
    position += 1

    matrix_size = number_of_points + 1
    distances = []
    for _ in range(matrix_size):
        distances.append(numbers[position:position + matrix_size])
        position += matrix_size

    return number_of_points, number_of_postmen, distances


def route_cost(route, distances):
    return sum(distances[first][second] for first, second in zip(route, route[1:]))


def route_costs(routes, distances):
    return [route_cost(route, distances) for route in routes]


def objective(costs):
    return max(costs) if costs else 0


def distribute_round_robin(nodes, number_of_postmen):
    routes = [[0] for _ in range(number_of_postmen)]
    for index, node in enumerate(nodes):
        routes[index % number_of_postmen].append(node)
    return routes


def initial_solution(number_of_points, number_of_postmen, distances):
    best_routes = None
    best_objective = INF
    seed_count = 8 if number_of_points <= 300 else 5

    for seed in range(RANDOM_SEED, RANDOM_SEED + seed_count):
        nodes = list(range(1, number_of_points + 1))
        random.Random(seed).shuffle(nodes)
        routes = distribute_round_robin(nodes, number_of_postmen)
        current_objective = objective(route_costs(routes, distances))
        if current_objective < best_objective:
            best_routes = routes
            best_objective = current_objective

    return best_routes


def remove_delta(route, position, distances):
    previous_node = route[position - 1]
    removed_node = route[position]
    if position == len(route) - 1:
        return -distances[previous_node][removed_node]

    next_node = route[position + 1]
    return (
        distances[previous_node][next_node]
        - distances[previous_node][removed_node]
        - distances[removed_node][next_node]
    )


def insert_delta(route, position, node, distances):
    previous_node = route[position - 1]
    if position == len(route):
        return distances[previous_node][node]

    next_node = route[position]
    return (
        distances[previous_node][node]
        + distances[node][next_node]
        - distances[previous_node][next_node]
    )


def best_insert_position(route, node, distances):
    best_position = 1
    best_delta = INF
    for position in range(1, len(route) + 1):
        delta = insert_delta(route, position, node, distances)
        if delta < best_delta:
            best_position = position
            best_delta = delta
    return best_position, best_delta


def ranked_cost_key(costs, changes):
    changed_costs = costs[:]
    for route_index, new_cost in changes:
        changed_costs[route_index] = new_cost
    return tuple(sorted(changed_costs, reverse=True)[:min(5, len(changed_costs))])


def source_routes(costs, limit):
    return sorted(range(len(costs)), key=lambda index: costs[index], reverse=True)[:limit]


def target_routes(costs, source_route, limit):
    candidates = [index for index in range(len(costs)) if index != source_route]
    return sorted(candidates, key=lambda index: costs[index])[:limit]


def candidate_positions(route, distances, limit):
    positions = []
    for position in range(1, len(route)):
        positions.append((remove_delta(route, position, distances), position))
    positions.sort()
    return [position for _, position in positions[:limit]]


def objective_after_change(costs, first_route, first_cost, second_route, second_cost):
    best = 0
    for route_index, cost in enumerate(costs):
        if route_index == first_route:
            value = first_cost
        elif route_index == second_route:
            value = second_cost
        else:
            value = cost
        best = max(best, value)
    return best


def best_relocate_move(routes, costs, distances, tabu, best_objective, iteration, parameters):
    best_move = None
    best_key = (INF, INF, INF)

    for source in source_routes(costs, parameters["source_routes"]):
        if len(routes[source]) <= 1:
            continue

        for position in candidate_positions(routes[source], distances, parameters["candidate_nodes"]):
            node = routes[source][position]
            new_source_cost = costs[source] + remove_delta(routes[source], position, distances)

            for target in target_routes(costs, source, parameters["target_routes"]):
                insert_position, target_delta = best_insert_position(routes[target], node, distances)
                new_target_cost = costs[target] + target_delta
                new_objective = objective_after_change(
                    costs,
                    source,
                    new_source_cost,
                    target,
                    new_target_cost,
                )

                tabu_key = ("relocate", node, source)
                if tabu.get(tabu_key, -1) > iteration and new_objective >= best_objective:
                    continue

                key = ranked_cost_key(costs, ((source, new_source_cost), (target, new_target_cost))) + (target_delta,)
                if key < best_key:
                    best_key = key
                    best_move = (
                        "relocate",
                        source,
                        target,
                        position,
                        insert_position,
                        node,
                        new_source_cost,
                        new_target_cost,
                    )

    return best_move


def random_route_with_nodes(routes, rng):
    candidates = [index for index, route in enumerate(routes) if len(route) > 1]
    return rng.choice(candidates) if candidates else None


def sampled_relocate_move(routes, costs, distances, tabu, best_objective, iteration, parameters, rng):
    best_move = None
    best_key = (INF, INF, INF)

    for _ in range(parameters["random_neighbors"]):
        source = random_route_with_nodes(routes, rng)
        if source is None:
            break
        target = rng.randrange(len(routes))
        if source == target:
            continue

        position = rng.randrange(1, len(routes[source]))
        node = routes[source][position]
        new_source_cost = costs[source] + remove_delta(routes[source], position, distances)
        insert_position, target_delta = best_insert_position(routes[target], node, distances)
        new_target_cost = costs[target] + target_delta
        new_objective = objective_after_change(costs, source, new_source_cost, target, new_target_cost)

        tabu_key = ("relocate", node, source)
        if tabu.get(tabu_key, -1) > iteration and new_objective >= best_objective:
            continue

        key = ranked_cost_key(costs, ((source, new_source_cost), (target, new_target_cost))) + (new_target_cost,)
        if key < best_key:
            best_key = key
            best_move = (
                "relocate",
                source,
                target,
                position,
                insert_position,
                node,
                new_source_cost,
                new_target_cost,
            )

    return best_move


def insert_positions(route, limit):
    if len(route) <= limit:
        return list(range(1, len(route) + 1))
    step = max(1, len(route) // limit)
    positions = list(range(1, len(route) + 1, step))
    if positions[-1] != len(route):
        positions.append(len(route))
    return positions[:limit]


def reinsert_cost(route, from_position, to_position, distances):
    adjusted_position = to_position - 1 if to_position > from_position else to_position
    if adjusted_position == from_position:
        return None, adjusted_position

    candidate = route[:]
    node = candidate.pop(from_position)
    candidate.insert(adjusted_position, node)
    return route_cost(candidate, distances), adjusted_position


def best_intra_move(routes, costs, distances, tabu, best_objective, iteration, parameters):
    best_move = None
    best_key = (INF, INF, INF)

    for route_index in source_routes(costs, parameters["intra_routes"]):
        route = routes[route_index]
        if len(route) <= 3:
            continue

        for from_position in candidate_positions(route, distances, parameters["intra_nodes"]):
            node = route[from_position]
            for to_position in insert_positions(route, parameters["intra_positions"]):
                new_cost, adjusted_position = reinsert_cost(route, from_position, to_position, distances)
                if new_cost is None:
                    continue

                new_objective = max(
                    new_cost if index == route_index else cost
                    for index, cost in enumerate(costs)
                )
                tabu_key = ("intra", node, route_index)
                if tabu.get(tabu_key, -1) > iteration and new_objective >= best_objective:
                    continue

                key = ranked_cost_key(costs, ((route_index, new_cost),))
                if key < best_key:
                    best_key = key
                    best_move = ("intra", route_index, from_position, adjusted_position, node, new_cost)

    return best_move


def choose_move(routes, costs, distances, tabu, best_objective, iteration, parameters, rng, stagnant):
    relocate = best_relocate_move(routes, costs, distances, tabu, best_objective, iteration, parameters)
    random_move = None
    if stagnant >= parameters["diversify_after"]:
        random_move = sampled_relocate_move(routes, costs, distances, tabu, best_objective, iteration, parameters, rng)
    intra = best_intra_move(routes, costs, distances, tabu, best_objective, iteration, parameters)
    moves = [move for move in (relocate, random_move, intra) if move is not None]
    if not moves:
        return None

    def key(move):
        if move[0] == "relocate":
            _, source, target, _, _, _, new_source_cost, new_target_cost = move
            return ranked_cost_key(costs, ((source, new_source_cost), (target, new_target_cost)))
        _, route_index, _, _, _, new_cost = move
        return ranked_cost_key(costs, ((route_index, new_cost),))

    return min(moves, key=key)


def apply_relocate(routes, costs, move):
    _, source, target, position, insert_position, node, new_source_cost, new_target_cost = move
    routes[source].pop(position)
    routes[target].insert(insert_position, node)
    costs[source] = new_source_cost
    costs[target] = new_target_cost
    return source, node


def apply_intra(routes, costs, move):
    _, route_index, from_position, to_position, node, new_cost = move
    routes[route_index].pop(from_position)
    routes[route_index].insert(to_position, node)
    costs[route_index] = new_cost
    return route_index, node


def parameters(number_of_points, number_of_postmen):
    if number_of_points <= 100:
        return {
            "iterations": 5400,
            "source_routes": min(4, number_of_postmen),
            "target_routes": min(8, number_of_postmen - 1),
            "candidate_nodes": 80,
            "random_neighbors": 120,
            "diversify_after": 120,
            "intra_routes": 1,
            "intra_nodes": 8,
            "intra_positions": 14,
            "tabu_tenure": 18,
        }
    if number_of_points <= 300:
        return {
            "iterations": 3600,
            "source_routes": min(4, number_of_postmen),
            "target_routes": min(8, number_of_postmen - 1),
            "candidate_nodes": 70,
            "random_neighbors": 140,
            "diversify_after": 120,
            "intra_routes": 1,
            "intra_nodes": 6,
            "intra_positions": 12,
            "tabu_tenure": 20,
        }
    if number_of_points <= 700:
        return {
            "iterations": 2400,
            "source_routes": min(3, number_of_postmen),
            "target_routes": min(7, number_of_postmen - 1),
            "candidate_nodes": 55,
            "random_neighbors": 160,
            "diversify_after": 150,
            "intra_routes": 1,
            "intra_nodes": 5,
            "intra_positions": 10,
            "tabu_tenure": 22,
        }
    return {
        "iterations": 1950,
        "source_routes": min(3, number_of_postmen),
        "target_routes": min(7, number_of_postmen - 1),
        "candidate_nodes": 45,
        "random_neighbors": 180,
        "diversify_after": 150,
        "intra_routes": 1,
        "intra_nodes": 4,
        "intra_positions": 10,
        "tabu_tenure": 24,
    }


def tabu_search(number_of_points, number_of_postmen, distances):
    if number_of_postmen <= 1:
        return [list(range(0, number_of_points + 1))]

    routes = initial_solution(number_of_points, number_of_postmen, distances)
    costs = route_costs(routes, distances)
    best_routes = [route[:] for route in routes]
    best_objective = objective(costs)
    config = parameters(number_of_points, number_of_postmen)
    tabu = {}
    rng = random.Random(RANDOM_SEED + number_of_points * 31 + number_of_postmen)
    stagnant = 0
    start_time = time.perf_counter()

    for iteration in range(config["iterations"]):
        if time.perf_counter() - start_time > TIME_LIMIT_SECONDS:
            break

        move = choose_move(routes, costs, distances, tabu, best_objective, iteration, config, rng, stagnant)
        if move is None:
            tabu.clear()
            continue

        if move[0] == "relocate":
            old_route, node = apply_relocate(routes, costs, move)
            tabu[("relocate", node, old_route)] = iteration + config["tabu_tenure"]
        else:
            route_index, node = apply_intra(routes, costs, move)
            tabu[("intra", node, route_index)] = iteration + config["tabu_tenure"]
        tabu = {key: expiry for key, expiry in tabu.items() if expiry > iteration}

        current_objective = objective(costs)
        if current_objective < best_objective:
            best_objective = current_objective
            best_routes = [route[:] for route in routes]
            stagnant = 0
        else:
            stagnant += 1

    return best_routes


def print_solution(routes):
    output_lines = [str(len(routes))]
    for route in routes:
        output_lines.append(str(len(route)))
        output_lines.append(" ".join(map(str, route)))
    sys.stdout.write("\n".join(output_lines))


def main():
    number_of_points, number_of_postmen, distances = read_input()
    print_solution(tabu_search(number_of_points, number_of_postmen, distances))


if __name__ == "__main__":
    main()

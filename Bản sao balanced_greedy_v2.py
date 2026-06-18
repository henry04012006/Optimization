import sys
import time
from pathlib import Path


TIME_LIMIT_MS = 30000.0


def parse_instance(text):
    numbers = list(map(int, text.split()))
    position = 0
    number_of_points = numbers[position]
    position += 1
    number_of_postmen = numbers[position]
    position += 1

    matrix_size = number_of_points + 1
    distances = []
    for _ in range(matrix_size):
        row = numbers[position:position + matrix_size]
        distances.append(row)
        position += matrix_size

    return number_of_points, number_of_postmen, distances


def parse_routes(text):
    numbers = list(map(int, text.split()))
    position = 0
    route_count = numbers[position]
    position += 1
    routes = []

    for _ in range(route_count):
        route_size = numbers[position]
        position += 1
        route = numbers[position:position + route_size]
        position += route_size
        routes.append(route)

    return routes


def find_best_balanced_assignment(routes, route_lengths, unvisited_nodes, distances, depot_distances):
    best_route_index = 0
    best_position = 0
    best_node = unvisited_nodes[0]
    first_added_distance = distances[routes[0][-1]][best_node]
    best_key = (
        route_lengths[0] + first_added_distance,
        first_added_distance,
        depot_distances[best_node],
        0,
        best_node,
    )

    for route_index, route in enumerate(routes):
        row = distances[route[-1]]
        current_length = route_lengths[route_index]

        for position, node in enumerate(unvisited_nodes):
            added_distance = row[node]
            key = (
                current_length + added_distance,
                added_distance,
                depot_distances[node],
                route_index,
                node,
            )
            if key < best_key:
                best_route_index = route_index
                best_position = position
                best_node = node
                best_key = key

    return best_route_index, best_position, best_node, best_key[1]


def remove_unvisited_node(unvisited_nodes, position):
    last_position = len(unvisited_nodes) - 1
    unvisited_nodes[position] = unvisited_nodes[last_position]
    unvisited_nodes.pop()


def solve(number_of_points, number_of_postmen, distances):
    routes = [[0] for _ in range(number_of_postmen)]
    route_lengths = [0] * number_of_postmen
    unvisited_nodes = list(range(1, number_of_points + 1))
    depot_distances = distances[0]

    while unvisited_nodes:
        route_index, position, next_node, added_distance = find_best_balanced_assignment(
            routes,
            route_lengths,
            unvisited_nodes,
            distances,
            depot_distances,
        )

        routes[route_index].append(next_node)
        route_lengths[route_index] += added_distance
        remove_unvisited_node(unvisited_nodes, position)

    return routes


def calculate_route_lengths(routes, distances):
    lengths = []
    for route in routes:
        route_length = 0
        for first_node, second_node in zip(route, route[1:]):
            route_length += distances[first_node][second_node]
        lengths.append(route_length)
    return lengths


def calculate_solution_cost(routes, distances):
    route_lengths = calculate_route_lengths(routes, distances)
    if not route_lengths:
        return 0
    return max(route_lengths)


def is_valid_solution(routes, number_of_points, number_of_postmen):
    if len(routes) != number_of_postmen:
        return False

    visited_nodes = []
    for route in routes:
        if not route or route[0] != 0:
            return False
        if any(node < 0 or node > number_of_points for node in route):
            return False
        visited_nodes.extend(route[1:])

    return sorted(visited_nodes) == list(range(1, number_of_points + 1))


def calculate_score(jury_solution, participant_solution, is_valid):
    if not is_valid or participant_solution <= 0:
        return 0.0
    return min(100.0, 100.0 * jury_solution / participant_solution)


def evaluate_test_case(input_path):
    output_path = input_path.with_suffix(".out")
    number_of_points, number_of_postmen, distances = parse_instance(input_path.read_text())
    jury_routes = parse_routes(output_path.read_text())

    start_time = time.perf_counter()
    participant_routes = solve(number_of_points, number_of_postmen, distances)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    jury_solution = calculate_solution_cost(jury_routes, distances)
    participant_solution = calculate_solution_cost(participant_routes, distances)
    valid = is_valid_solution(participant_routes, number_of_points, number_of_postmen)
    score = calculate_score(jury_solution, participant_solution, valid)
    printed_score = int(score)
    status = "TLE" if elapsed_ms > TIME_LIMIT_MS else "OK"

    return {
        "test": input_path.stem,
        "points": number_of_points,
        "postmen": number_of_postmen,
        "jury": jury_solution,
        "participant": participant_solution,
        "score": score,
        "printed_score": printed_score,
        "runtime_ms": elapsed_ms,
        "valid": valid,
        "status": status,
    }


def print_report(results):
    header = (
        f"{'test':<10} {'N':>5} {'K':>4} {'jury':>6} "
        f"{'participant':>12} {'score':>7} {'runtime(ms)':>12} {'valid':>7} {'status':>7}"
    )
    print(header)
    print("-" * len(header))

    total_score = 0
    total_runtime_ms = 0.0
    for result in results:
        total_score += result["printed_score"]
        total_runtime_ms += result["runtime_ms"]
        print(
            f"{result['test']:<10} "
            f"{result['points']:>5} "
            f"{result['postmen']:>4} "
            f"{result['jury']:>6} "
            f"{result['participant']:>12} "
            f"{result['printed_score']:>7} "
            f"{result['runtime_ms']:>12.2f} "
            f"{str(result['valid']):>7} "
            f"{result['status']:>7}"
        )

    if results:
        average_score = total_score // len(results)
        print("-" * len(header))
        print(f"{'total':<10} {'':>5} {'':>4} {'':>6} {'':>12} {total_score:>7} {total_runtime_ms:>12.2f} {'':>7} {'':>7}")
        print(f"{'average':<10} {'':>5} {'':>4} {'':>6} {'':>12} {average_score:>7} {'':>12} {'':>7} {'':>7}")


def main():
    tests_dir = Path("tests")
    input_paths = sorted(tests_dir.glob("*.in"))

    if not input_paths:
        sys.exit("No .in files found in tests/")

    results = [evaluate_test_case(input_path) for input_path in input_paths]
    print_report(results)


if __name__ == "__main__":
    main()

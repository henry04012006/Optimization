import random
import sys
import time
from pathlib import Path

from tabu_searchv3 import objective, route_costs, tabu_search


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
        distances.append(numbers[position:position + matrix_size])
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
        routes.append(numbers[position:position + route_size])
        position += route_size
    return routes


def valid_solution(routes, number_of_points, number_of_postmen):
    if len(routes) != number_of_postmen:
        return False

    visited = []
    for route in routes:
        if not route or route[0] != 0:
            return False
        if any(node < 0 or node > number_of_points for node in route):
            return False
        visited.extend(route[1:])
    return sorted(visited) == list(range(1, number_of_points + 1))


def score(jury_solution, participant_solution, valid):
    if not valid or participant_solution <= 0:
        return 0
    return int(min(100.0, 100.0 * jury_solution / participant_solution))


def evaluate_test(input_path):
    output_path = input_path.with_suffix(".out")
    number_of_points, number_of_postmen, distances = parse_instance(input_path.read_text())
    jury_routes = parse_routes(output_path.read_text())

    start_time = time.perf_counter()
    participant_routes = tabu_search(number_of_points, number_of_postmen, distances)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    jury_solution = objective(route_costs(jury_routes, distances))
    participant_solution = objective(route_costs(participant_routes, distances))
    valid = valid_solution(participant_routes, number_of_points, number_of_postmen)
    return {
        "test": input_path.stem,
        "points": number_of_points,
        "postmen": number_of_postmen,
        "jury": jury_solution,
        "participant": participant_solution,
        "score": score(jury_solution, participant_solution, valid),
        "runtime_ms": elapsed_ms,
        "valid": valid,
        "status": "TLE" if elapsed_ms > TIME_LIMIT_MS else "OK",
    }


def print_report(results):
    header = (
        f"{'test':<10} {'N':>5} {'K':>4} {'jury':>6} "
        f"{'participant':>12} {'score':>7} {'runtime(ms)':>12} {'valid':>7} {'status':>7}"
    )
    print(header)
    print("-" * len(header))

    total_score = 0
    total_runtime = 0.0
    for result in results:
        total_score += result["score"]
        total_runtime += result["runtime_ms"]
        print(
            f"{result['test']:<10} "
            f"{result['points']:>5} "
            f"{result['postmen']:>4} "
            f"{result['jury']:>6} "
            f"{result['participant']:>12} "
            f"{result['score']:>7} "
            f"{result['runtime_ms']:>12.2f} "
            f"{str(result['valid']):>7} "
            f"{result['status']:>7}"
        )

    if results:
        print("-" * len(header))
        print(f"{'total':<10} {'':>5} {'':>4} {'':>6} {'':>12} {total_score:>7} {total_runtime:>12.2f} {'':>7} {'':>7}")
        print(f"{'average':<10} {'':>5} {'':>4} {'':>6} {'':>12} {total_score // len(results):>7} {'':>12} {'':>7} {'':>7}")


def main():
    input_paths = sorted(Path("tests").glob("*.in"))
    if not input_paths:
        sys.exit("No .in files found in tests/")
    print_report([evaluate_test(input_path) for input_path in input_paths])


if __name__ == "__main__":
    main()

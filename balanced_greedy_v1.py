import sys


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
        row = numbers[position:position + matrix_size]
        distances.append(row)
        position += matrix_size

    return number_of_points, number_of_postmen, distances


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


def print_solution(routes):
    output_lines = [str(len(routes))]
    for route in routes:
        output_lines.append(str(len(route)))
        output_lines.append(" ".join(map(str, route)))
    sys.stdout.write("\n".join(output_lines))


def main():
    number_of_points, number_of_postmen, distances = read_input()
    routes = solve(number_of_points, number_of_postmen, distances)
    print_solution(routes)


if __name__ == "__main__":
    main()

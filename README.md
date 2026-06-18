# Min-Max Vehicle Routing Problem with Postman Collecting Packages

## 1. Problem Overview

The Vehicle Routing Problem is a fundamental optimization problem in logistics and transportation. In this problem, multiple postmen start from a post office and collect packages from a set of pickup points.

This project focuses on a specific variant called the **Min-Max Vehicle Routing Problem with Postman Collecting Packages**.

There are:

* `N` pickup points, indexed from `1` to `N`.
* `K` postmen, all starting from the post office.
* The post office is denoted as node `0`.
* A distance matrix `d(i, j)` is given for all pairs of nodes `i, j`, where `i, j ∈ {0, 1, ..., N}`.

Each postman is assigned one route. A route starts from the depot `0` and then visits several pickup points in a specific order. Every pickup point must be collected exactly once by exactly one postman.

The goal is to build `K` routes such that the longest route among all postmen is as short as possible.

In other words, the objective is not only to reduce travel distance, but also to balance the workload between postmen.

---

## 2. Problem Characteristics

This problem contains two main decisions:

1. **Assignment decision**

   Each pickup point must be assigned to one postman.

2. **Sequencing decision**

   For each postman, the assigned pickup points must be arranged in a visiting order.

Because both assignment and ordering are involved, the number of possible solutions grows very quickly when `N` becomes large. Therefore, exhaustive search is impractical for large test cases.

---

## 3. Mathematical Definition

Let:

* `N` be the number of pickup points.
* `K` be the number of postmen.
* `V = {0, 1, 2, ..., N}` be the set of all nodes.
* `0` be the depot / post office.
* `P = {1, 2, ..., K}` be the set of postmen.
* `d(i, j)` be the distance from node `i` to node `j`.

A solution consists of `K` routes.

For postman `k`, the route is represented as:

```text
Rk = (x1, x2, ..., xl)
```

where:

* `l` is the number of nodes in the route.
* `x1 = 0`, meaning every route starts from the depot.
* `x2, x3, ..., xl` are pickup points assigned to postman `k`.

The length of a route is the total distance traveled along its sequence:

```text
Lk = d(x1, x2) + d(x2, x3) + ... + d(x(l-1), xl)
```

The objective is:

```text
Minimize max(L1, L2, ..., LK)
```

This means the solution should minimize the maximum route length among all postmen.

---

## 4. Required Constraints

A valid solution must satisfy the following constraints.

### 4.1 Every pickup point must be visited

Each pickup point from `1` to `N` must appear in one route.

```text
Every node in {1, 2, ..., N} must be visited.
```

### 4.2 Each pickup point must be visited exactly once

No pickup point can appear in more than one route.

```text
Each pickup point belongs to exactly one postman's route.
```

### 4.3 Every route must start from the depot

For every postman route:

```text
x1 = 0
```

### 4.4 There must be exactly K routes

The output must contain exactly `K` routes, one for each postman.

A route may contain only the depot if no pickup point is assigned to that postman, depending on the solution strategy and test case requirements.

---

## 5. Important Note About Returning to Depot

The problem statement defines each route as starting from the depot.

The official output format and sample output represent each route as:

```text
0 pickup_1 pickup_2 ...
```

The sample output does not explicitly include a final `0` at the end of each route.

Therefore, for output formatting, each route should start with `0`, followed by the pickup points assigned to that postman.

If the route length calculation internally needs to include the distance back to the depot, that should be handled inside the algorithm. However, the printed route should follow the required output format.

---

## 6. Input Format

The first line contains two integers:

```text
N K
```

where:

```text
1 <= N <= 1000
1 <= K <= 100
```

The next `N + 1` lines contain the distance matrix.

Line `i + 1`, with `i = 0, 1, ..., N`, contains the `i`-th row of the matrix:

```text
d(i, 0) d(i, 1) d(i, 2) ... d(i, N)
```

The matrix size is:

```text
(N + 1) x (N + 1)
```

The distance matrix may contain zero values. Do not assume that `d(i, j) > 0` for all `i != j`.

The distance matrix may also be asymmetric unless the problem explicitly states otherwise. Therefore, do not assume:

```text
d(i, j) = d(j, i)
```

---

## 7. Output Format

The output must describe the `K` routes.

The first line contains:

```text
K
```

Then, for each postman `k = 1, 2, ..., K`, print two lines.

The first line contains the number of nodes in the route:

```text
lk
```

The second line contains `lk` integers representing the route sequence:

```text
x1 x2 ... xlk
```

where:

```text
x1 = 0
```

and the remaining elements are pickup points.

The elements in one route must be separated by a single space.

---

## 8. Example

### Input

```text
6 2
0 9 9 9 7 2 9
9 0 3 0 2 8 1
9 3 0 3 4 7 4
9 0 3 0 2 8 1
7 2 4 2 0 6 2
2 8 7 8 6 0 8
9 1 4 1 2 8 0
```

### Output

```text
2
3
0 5 2
5
0 4 1 3 6
```

In this output:

* There are `2` postmen.
* Route 1 is:

```text
0 -> 5 -> 2
```

* Route 2 is:

```text
0 -> 4 -> 1 -> 3 -> 6
```

All pickup points from `1` to `6` appear exactly once.

---

## 9. Valid Solution Requirements

A generated solution is considered valid if:

1. The first line of output is exactly `K`.
2. Exactly `K` routes are printed.
3. Each route has two lines:

   * one line for `lk`;
   * one line for the route sequence.
4. Each route starts with node `0`.
5. Every pickup point from `1` to `N` appears exactly once across all routes.
6. No pickup point is missing.
7. No pickup point is duplicated.
8. No node outside the range `0` to `N` appears in the output.
9. The number `lk` must match the number of integers printed in the corresponding route line.

---

## 10. Optimization Objective

Among all valid solutions, the goal is to minimize the maximum route length.

For each postman `k`, compute the route length:

```text
Lk = sum of distances between consecutive nodes in route k
```

The cost of the whole solution is:

```text
max(L1, L2, ..., LK)
```

The objective is:

```text
minimize max(L1, L2, ..., LK)
```

This objective encourages balanced routes and avoids assigning one postman a route that is much longer than the others.

---

## 11. Implementation Expectations

The program should:

1. Read input from standard input.
2. Construct a valid set of `K` routes.
3. Ensure that every pickup point is collected exactly once.
4. Output the routes in the required format.
5. Try to minimize the longest route among all postmen.

The program should not hardcode the sample input or output.

The program should handle large input sizes up to:

```text
N = 1000
K = 100
```

Therefore, memory usage and runtime should be considered carefully.
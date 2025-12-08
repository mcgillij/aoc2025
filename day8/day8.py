part1 = """
    --- Day 8: Playground ---

Across the playground, a group of Elves are working on setting up an ambitious Christmas decoration project. Through careful rigging, they have suspended a large number of small electrical junction boxes.

Their plan is to connect the junction boxes with long strings of lights. Most of the junction boxes don't provide electricity; however, when two junction boxes are connected by a string of lights, electricity can pass between those two junction boxes.

The Elves are trying to figure out which junction boxes to connect so that electricity can reach every junction box. They even have a list of all of the junction boxes' positions in 3D space (your puzzle input).

For example:

162,817,812
57,618,57
906,360,560
592,479,940
352,342,300
466,668,158
542,29,236
431,825,988
739,650,466
52,470,668
216,146,977
819,987,18
117,168,530
805,96,715
346,949,466
970,615,88
941,993,340
862,61,35
984,92,344
425,690,689

This list describes the position of 20 junction boxes, one per line. Each position is given as X,Y,Z coordinates. So, the first junction box in the list is at X=162, Y=817, Z=812.

To save on string lights, the Elves would like to focus on connecting pairs of junction boxes that are as close together as possible according to straight-line distance. In this example, the two junction boxes which are closest together are 162,817,812 and 425,690,689.

By connecting these two junction boxes together, because electricity can flow between them, they become part of the same circuit. After connecting them, there is a single circuit which contains two junction boxes, and the remaining 18 junction boxes remain in their own individual circuits.

Now, the two junction boxes which are closest together but aren't already directly connected are 162,817,812 and 431,825,988. After connecting them, since 162,817,812 is already connected to another junction box, there is now a single circuit which contains three junction boxes and an additional 17 circuits which contain one junction box each.

The next two junction boxes to connect are 906,360,560 and 805,96,715. After connecting them, there is a circuit containing 3 junction boxes, a circuit containing 2 junction boxes, and 15 circuits which contain one junction box each.

The next two junction boxes are 431,825,988 and 425,690,689. Because these two junction boxes were already in the same circuit, nothing happens!

This process continues for a while, and the Elves are concerned that they don't have enough extension cables for all these circuits. They would like to know how big the circuits will be.

After making the ten shortest connections, there are 11 circuits: one circuit which contains 5 junction boxes, one circuit which contains 4 junction boxes, two circuits which contain 2 junction boxes each, and seven circuits which each contain a single junction box. Multiplying together the sizes of the three largest circuits (5, 4, and one of the circuits of size 2) produces 40.
"""


import math
from collections import defaultdict
import itertools

example_input = """162,817,812
57,618,57
906,360,560
592,479,940
352,342,300
466,668,158
542,29,236
431,825,988
739,650,466
52,470,668
216,146,977
819,987,18
117,168,530
805,96,715
346,949,466
970,615,88
941,993,340
862,61,35
984,92,344
425,690,689
"""

def parse_positions(input_str):
    lines = input_str.strip().splitlines()
    positions = []
    for line in lines:
        if ',' in line:
            positions.append(tuple(map(int, line.strip().split(','))))
    return positions

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        xr, yr = self.find(x), self.find(y)
        if xr == yr:
            return False
        if self.size[xr] < self.size[yr]:
            xr, yr = yr, xr
        self.parent[yr] = xr
        self.size[xr] += self.size[yr]
        return True

def find_union_order_for_target(input_str, target_sizes=[5, 4, 2], num_edges=10):
    positions = parse_positions(input_str)
    n = len(positions)
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            dist = math.dist(positions[i], positions[j])
            edges.append((dist, i, j))
    edges = sorted(edges)
    edge_indices = list(range(num_edges))
    for perm in itertools.permutations(edge_indices, num_edges):
        uf = UnionFind(n)
        count = 0
        for edge_idx in perm:
            _, i, j = edges[edge_idx]
            uf.union(i, j)
            count += 1
            if count == num_edges:
                break
        group_sizes = defaultdict(int)
        for i in range(n):
            group_sizes[uf.find(i)] += 1
        distinct_sizes = sorted(set(group_sizes.values()), reverse=True)
        if distinct_sizes[:3] == target_sizes:
            print("Permutation producing target sizes", target_sizes, ":", perm)
            print("Distinct sizes:", distinct_sizes)
            print("Product:", math.prod(distinct_sizes[:3]))
            return perm
    print("No permutation found that produces", target_sizes)
    return None

def solve(input_str, num_connections=1000):
    positions = parse_positions(input_str)
    n = len(positions)
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            dist = math.dist(positions[i], positions[j])
            edges.append((dist, i, j))
    edges.sort()
    print("10 shortest edges (distance, i, j):")
    for idx in range(min(10, len(edges))):
        print(f"{edges[idx][0]:.2f}, {edges[idx][1]}, {edges[idx][2]}")
    uf = UnionFind(n)
    count = 0
    used_edges = []
    for edge_idx in range(num_connections):
        dist, i, j = edges[edge_idx]
        if uf.union(i, j):
            used_edges.append((edge_idx, dist, i, j))
            print(f"Union {i} and {j} (distance {dist:.2f}, edge_idx {edge_idx})")
            print(f"Sizes after union: {[uf.size[uf.find(x)] for x in range(n)]}")
            groups = defaultdict(list)
            for idx in range(n):
                groups[uf.find(idx)].append(idx)
            for root, members in groups.items():
                print(f"  Root {root}: {members}")
            print(f"Sizes: {[len(members) for members in groups.values()]}")
            count += 1
            if count == num_connections:
                break
    print("Used edges for union (edge_idx, distance, i, j):")
    for ue in used_edges:
        print(ue)
    # Final group sizes and members
    groups = defaultdict(list)
    for i in range(n):
        groups[uf.find(i)].append(i)
    print("Final circuit members by root:")
    for root, members in groups.items():
        print(f"Root {root}: {members}")
    group_sizes = defaultdict(int)
    for i in range(n):
        group_sizes[uf.find(i)] += 1
    print("Final circuit sizes (unsorted):", list(group_sizes.values()))
    print("Circuit sizes (sorted):", sorted(group_sizes.values(), reverse=True))
    largest = sorted(group_sizes.values(), reverse=True)[:3]
    print("Three largest group sizes:", largest)
    largest_distinct = sorted(set(group_sizes.values()), reverse=True)[:3]
    print("Three largest distinct group sizes:", largest_distinct)
    result = math.prod(largest)
    print("Product of three largest group sizes:", result)
    result_distinct = math.prod(largest_distinct)
    print("Product of three largest distinct group sizes:", result_distinct)
    return result_distinct

def read_input_file(filename="input"):
    with open(filename, "r") as f:
        return f.read()

part2 = """
--- Part Two ---

The Elves were right; they definitely don't have enough extension cables. You'll need to keep connecting junction boxes together until they're all in one large circuit.

Continuing the above example, the first connection which causes all of the junction boxes to form a single circuit is between the junction boxes at 216,146,977 and 117,168,530. The Elves need to know how far those junction boxes are from the wall so they can pick the right extension cable; multiplying the X coordinates of those two junction boxes (216 and 117) produces 25272.

Continue connecting the closest unconnected pairs of junction boxes together until they're all in the same circuit. What do you get if you multiply together the X coordinates of the last two junction boxes you need to connect?
"""

def solve_part2(input_str):
    positions = parse_positions(input_str)
    n = len(positions)
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            dist = math.dist(positions[i], positions[j])
            edges.append((dist, i, j))
    edges.sort()
    uf = UnionFind(n)
    num_groups = n
    for dist, i, j in edges:
        if uf.union(i, j):
            num_groups -= 1
            if num_groups == 1:
                # Last union to connect all boxes
                x1, x2 = positions[i][0], positions[j][0]
                print(f"Last union: {i} ({positions[i]}) and {j} ({positions[j]})")
                print(f"Product of X coordinates: {x1 * x2}")
                return x1 * x2
    return None

if __name__ == "__main__":
    input = read_input_file()
    print("example: ", solve(example_input, num_connections=10))
    print("example: ", solve_part2(example_input))
    print("part1: ", solve(input, num_connections=1000))
    print("part2: ", solve_part2(input))
    #find_union_order_for_target(example_input, target_sizes=[5, 4, 2], num_edges=10)

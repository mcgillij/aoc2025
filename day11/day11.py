part1 = """--- Day 11: Reactor ---

You hear some loud beeping coming from a hatch in the floor of the factory, so you decide to check it out. Inside, you find several large electrical conduits and a ladder.

Climbing down the ladder, you discover the source of the beeping: a large, toroidal reactor which powers the factory above. Some Elves here are hurriedly running between the reactor and a nearby server rack, apparently trying to fix something.

One of the Elves notices you and rushes over. "It's a good thing you're here! We just installed a new server rack, but we aren't having any luck getting the reactor to communicate with it!" You glance around the room and see a tangle of cables and devices running from the server rack to the reactor. She rushes off, returning a moment later with a list of the devices and their outputs (your puzzle input).

For example:

aaa: you hhh
you: bbb ccc
bbb: ddd eee
ccc: ddd eee fff
ddd: ggg
eee: out
fff: out
ggg: out
hhh: ccc fff iii
iii: out

Each line gives the name of a device followed by a list of the devices to which its outputs are attached. So, bbb: ddd eee means that device bbb has two outputs, one leading to device ddd and the other leading to device eee.

The Elves are pretty sure that the issue isn't due to any specific device, but rather that the issue is triggered by data following some specific path through the devices. Data only ever flows from a device through its outputs; it can't flow backwards.

After dividing up the work, the Elves would like you to focus on the devices starting with the one next to you (an Elf hastily attaches a label which just says you) and ending with the main output to the reactor (which is the device with the label out).

To help the Elves figure out which path is causing the issue, they need you to find every path from you to out.

In this example, these are all of the paths from you to out:

    Data could take the connection from you to bbb, then from bbb to ddd, then from ddd to ggg, then from ggg to out.
    Data could take the connection to bbb, then to eee, then to out.
    Data could go to ccc, then ddd, then ggg, then out.
    Data could go to ccc, then eee, then out.
    Data could go to ccc, then fff, then out.

In total, there are 5 different paths leading from you to out."""

part2 = """--- Part Two ---

Thanks in part to your analysis, the Elves have figured out a little bit about the issue. They now know that the problematic data path passes through both dac (a digital-to-analog converter) and fft (a device which performs a fast Fourier transform).

They're still not sure which specific path is the problem, and so they now need you to find every path from svr (the server rack) to out. However, the paths you find must all also visit both dac and fft (in any order).

For example:

svr: aaa bbb
aaa: fft
fft: ccc
bbb: tty
tty: ccc
ccc: ddd eee
ddd: hub
hub: fff
eee: dac
dac: fff
fff: ggg hhh
ggg: out
hhh: out

This new list of devices contains many paths from svr to out:

svr,aaa,fft,ccc,ddd,hub,fff,ggg,out
svr,aaa,fft,ccc,ddd,hub,fff,hhh,out
svr,aaa,fft,ccc,eee,dac,fff,ggg,out
svr,aaa,fft,ccc,eee,dac,fff,hhh,out
svr,bbb,tty,ccc,ddd,hub,fff,ggg,out
svr,bbb,tty,ccc,ddd,hub,fff,hhh,out
svr,bbb,tty,ccc,eee,dac,fff,ggg,out
svr,bbb,tty,ccc,eee,dac,fff,hhh,out

However, only 2 paths from svr to out visit both dac and fft.

Find all of the paths that lead from svr to out. How many of those paths visit both dac and fft?
"""

example_input = """aaa: you hhh
you: bbb ccc
bbb: ddd eee
ccc: ddd eee fff
ddd: ggg
eee: out
fff: out
ggg: out
hhh: ccc fff iii
iii: out"""

part2_example_input = """svr: aaa bbb
aaa: fft
fft: ccc
bbb: tty
tty: ccc
ccc: ddd eee
ddd: hub
hub: fff
eee: dac
dac: fff
fff: ggg hhh
ggg: out
hhh: out"""

def parse_input(input_str):
    graph = {}
    all_nodes = set()
    for line in input_str.strip().splitlines():
        if ':' in line:
            node, outs = line.split(':')
            node = node.strip()
            neighbors = [x.strip() for x in outs.strip().split()]
            graph[node] = neighbors
            all_nodes.add(node)
            all_nodes.update(neighbors)
    # Ensure all nodes are present in the graph
    for n in all_nodes:
        if n not in graph:
            graph[n] = []
    return graph

def parse_input_file(filename="input"):
    with open(filename, "r") as f:
        content = f.read()
    return parse_input(content)

def paths_with_nodes(paths, required_nodes):
    return [p for p in paths if all(node in p for node in required_nodes)]

def find_paths(graph, start, end, path=None):
    if path is None:
        path = []
    path = path + [start]
    if start == end:
        return [path]
    if start not in graph:
        return []
    paths = []
    for node in graph[start]:
        if node not in path:
            newpaths = find_paths(graph, node, end, path)
            for newpath in newpaths:
                paths.append(newpath)
    return paths

def count_paths_with_nodes(graph, start, end, required_nodes):
    from functools import lru_cache
    # Prune graph to nodes that can reach end to reduce state space
    rev = {n: [] for n in graph}
    for u, outs in graph.items():
        for v in outs:
            rev[v].append(u)
    can_reach = set()
    stack = [end]
    while stack:
        v = stack.pop()
        if v in can_reach:
            continue
        can_reach.add(v)
        for p in rev.get(v, ()): 
            if p not in can_reach:
                stack.append(p)

    if start not in can_reach:
        return 0

    # Assign each reachable node a compact index for bitmasking
    reachable_nodes = [n for n in graph if n in can_reach]
    node_indices = {node: i for i, node in enumerate(reachable_nodes)}
    required_indices = {node: idx for idx, node in enumerate(required_nodes)}
    end_idx = node_indices[end]
    start_idx = node_indices[start]
    # build adjacency by index for faster iteration
    adj_idx = {node_indices[u]: [node_indices[v] for v in graph[u] if v in node_indices] for u in reachable_nodes}

    @lru_cache(maxsize=None)
    def dfs(node_idx, required_mask, visited_mask):
        if node_idx == end_idx:
            return int(required_mask == (1 << len(required_nodes)) - 1)
        total = 0
        for n_idx in adj_idx[node_idx]:
            if not (visited_mask & (1 << n_idx)):
                new_required_mask = required_mask
                # neighbor name from reachable_nodes list
                neighbor_name = reachable_nodes[n_idx]
                if neighbor_name in required_indices:
                    new_required_mask |= (1 << required_indices[neighbor_name])
                total += dfs(n_idx, new_required_mask, visited_mask | (1 << n_idx))
        return total

    return dfs(start_idx, 0, 1 << start_idx)


def count_paths_with_nodes_dag(graph, start, end, required_nodes):
    """Count simple paths from start to end visiting all required_nodes in a DAG.
    Uses DP over topological order with masks for required nodes."""
    # Map required nodes to bit positions
    req_list = list(required_nodes)
    req_index = {n: i for i, n in enumerate(req_list)}
    full_mask = (1 << len(req_list)) - 1

    # Prune graph to nodes that can reach end
    rev = {n: [] for n in graph}
    for u, outs in graph.items():
        for v in outs:
            rev[v].append(u)
    can_reach = set()
    stack = [end]
    while stack:
        v = stack.pop()
        if v in can_reach:
            continue
        can_reach.add(v)
        for p in rev.get(v, ()): 
            if p not in can_reach:
                stack.append(p)

    if start not in can_reach:
        return 0

    # Topological order on the reachable subgraph
    visited = set()
    topo = []

    def dfs_topo(u):
        visited.add(u)
        for v in graph[u]:
            if v in can_reach and v not in visited:
                dfs_topo(v)
        topo.append(u)

    dfs_topo(start)

    # DP table: for each node, an array of size 2^k counts
    dp = {n: [0] * (1 << len(req_list)) for n in topo}

    for u in topo:
        if u == end and len(graph[u]) == 0:
            base = (1 << req_index[u]) if u in req_index else 0
            dp[u][base] = 1

    # Process nodes in topological order, starting from sinks upwards
    for u in topo:
        base = (1 << req_index[u]) if u in req_index else 0
        # If u is end and already initialized, still aggregate neighbors (no outs for typical 'out')
        for v in graph[u]:
            if v not in can_reach:
                continue
            for mask_v, cnt in enumerate(dp.get(v, [])):
                if cnt:
                    dp[u][base | mask_v] += cnt

    return dp[start][full_mask]


def is_dag(graph):
    # Kahn's algorithm to detect cycles; returns True if DAG
    indeg = {n: 0 for n in graph}
    for u, outs in graph.items():
        for v in outs:
            indeg[v] = indeg.get(v, 0) + 1
    q = [n for n, d in indeg.items() if d == 0]
    idx = 0
    while idx < len(q):
        u = q[idx]
        idx += 1
        for v in graph[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return len(q) == len(graph)

if __name__ == "__main__":
    # Part 1 Example
    example_graph = parse_input(example_input)
    all_paths = find_paths(example_graph, "you", "out")
    print(f"Example: Total paths from 'you' to 'out': {len(all_paths)}")
    for p in all_paths:
        print(" -> ".join(p))

    # Part 2 Example
    part2_example_graph = parse_input(part2_example_input)
    all_part2_paths = find_paths(part2_example_graph, "svr", "out")
    filtered_paths = paths_with_nodes(all_part2_paths, ["dac", "fft"])
    print(f"Part2 Example: Total paths from 'svr' to 'out': {len(all_part2_paths)}")
    print(f"Part2 Example: Paths visiting both 'dac' and 'fft': {len(filtered_paths)}")
    for p in filtered_paths:
        print(" -> ".join(p))
    # Use optimized DAG algorithm when possible
    if is_dag(part2_example_graph):
        part2_example_count = count_paths_with_nodes_dag(part2_example_graph, "svr", "out", ["dac", "fft"])
    else:
        part2_example_count = count_paths_with_nodes(part2_example_graph, "svr", "out", ["dac", "fft"])
    print(f"Part2 Example (Optimized): Paths visiting both 'dac' and 'fft': {part2_example_count}")

    # Part 1 Actual Input
    input_graph = parse_input_file()
    all_paths = find_paths(input_graph, "you", "out")
    print(f"Part1: Total paths from 'you' to 'out': {len(all_paths)}")

    # Part 2 Actual Input
    if is_dag(input_graph):
        part2_count = count_paths_with_nodes_dag(input_graph, "svr", "out", ["dac", "fft"])
    else:
        part2_count = count_paths_with_nodes(input_graph, "svr", "out", ["dac", "fft"])
    print(f"Part2: Paths visiting both 'dac' and 'fft': {part2_count}")
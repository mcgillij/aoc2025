part1 = """--- Day 12: Christmas Tree Farm ---

You're almost out of time, but there can't be much left to decorate. Although there are no stairs, elevators, escalators, tunnels, chutes, teleporters, firepoles, or conduits here that would take you deeper into the North Pole base, there is a ventilation duct. You jump in.

After bumping around for a few minutes, you emerge into a large, well-lit cavern full of Christmas trees!

There are a few Elves here frantically decorating before the deadline. They think they'll be able to finish most of the work, but the one thing they're worried about is the presents for all the young Elves that live here at the North Pole. It's an ancient tradition to put the presents under the trees, but the Elves are worried they won't fit.

The presents come in a few standard but very weird shapes. The shapes and the regions into which they need to fit are all measured in standard units. To be aesthetically pleasing, the presents need to be placed into the regions in a way that follows a standardized two-dimensional unit grid; you also can't stack presents.

As always, the Elves have a summary of the situation (your puzzle input) for you. First, it contains a list of the presents' shapes. Second, it contains the size of the region under each tree and a list of the number of presents of each shape that need to fit into that region. For example:

0:
###
##.
##.

1:
###
##.
.##

2:
.##
###
##.

3:
##.
###
##.

4:
###
#..
###

5:
###
.#.
###

4x4: 0 0 0 0 2 0
12x5: 1 0 1 0 2 2
12x5: 1 0 1 0 3 2

The first section lists the standard present shapes. For convenience, each shape starts with its index and a colon; then, the shape is displayed visually, where # is part of the shape and . is not.

The second section lists the regions under the trees. Each line starts with the width and length of the region; 12x5 means the region is 12 units wide and 5 units long. The rest of the line describes the presents that need to fit into that region by listing the quantity of each shape of present; 1 0 1 0 3 2 means you need to fit one present with shape index 0, no presents with shape index 1, one present with shape index 2, no presents with shape index 3, three presents with shape index 4, and two presents with shape index 5.

Presents can be rotated and flipped as necessary to make them fit in the available space, but they have to always be placed perfectly on the grid. Shapes can't overlap (that is, the # part from two different presents can't go in the same place on the grid), but they can fit together (that is, the . part in a present's shape's diagram does not block another present from occupying that space on the grid).

The Elves need to know how many of the regions can fit the presents listed. In the above example, there are six unique present shapes and three regions that need checking.

The first region is 4x4:

....
....
....
....

In it, you need to determine whether you could fit two presents that have shape index 4:

###
#..
###

After some experimentation, it turns out that you can fit both presents in this region. Here is one way to do it, using A to represent one present and B to represent the other:

AAA.
ABAB
ABAB
.BBB

The second region, 12x5: 1 0 1 0 2 2, is 12 units wide and 5 units long. In that region, you need to try to fit one present with shape index 0, one present with shape index 2, two presents with shape index 4, and two presents with shape index 5.

It turns out that these presents can all fit in this region. Here is one way to do it, again using different capital letters to represent all the required presents:

....AAAFFE.E
.BBBAAFFFEEE
DDDBAAFFCECE
DBBB....CCC.
DDD.....C.C.

The third region, 12x5: 1 0 1 0 3 2, is the same size as the previous region; the only difference is that this region needs to fit one additional present with shape index 4. Unfortunately, no matter how hard you try, there is no way to fit all of the presents into this region.

So, in this example, 2 regions can fit all of their listed presents.

Consider the regions beneath each tree and the presents the Elves would like to fit into each of them. How many of the regions can fit all of the presents listed?
"""

example_input = """0:
###
##.
##.

1:
###
##.
.##

2:
.##
###
##.

3:
##.
###
##.

4:
###
#..
###

5:
###
.#.
###

4x4: 0 0 0 0 2 0
12x5: 1 0 1 0 2 2
12x5: 1 0 1 0 3 2
"""

import re
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

def parse_input(text):
    shapes = {}
    regions = []
    lines = text.strip().splitlines()
    i = 0
    while i < len(lines):
        if re.match(r'^\d+:$', lines[i]):
            idx = int(lines[i][:-1])
            i += 1
            shape = []
            while i < len(lines) and lines[i] and not lines[i].endswith(':'):
                shape.append(lines[i])
                i += 1
            shapes[idx] = shape
        elif re.match(r'^\d+x\d+:', lines[i]):
            m = re.match(r'^(\d+)x(\d+): (.+)$', lines[i])
            w, h = int(m.group(1)), int(m.group(2))
            counts = list(map(int, m.group(3).split()))
            regions.append((w, h, counts))
            i += 1
        else:
            i += 1
    return shapes, regions

def shape_variants(shape):
    """Generate all unique rotations and flips of a shape."""
    grids = set()
    grid = tuple(tuple(row) for row in shape)
    for _ in range(4):
        grid = tuple(zip(*grid[::-1]))  # rotate 90
        grids.add(grid)
        grids.add(tuple(tuple(row[::-1]) for row in grid))  # flip
    return [g for g in grids]

def compute_placements_for_size(shape_dict, W, H):
    """Return placements dict and areas for each shape for the given W,H."""
    placements = {}
    areas = {}
    total_cells = W * H
    for idx, variants in shape_dict.items():
        masks = set()
        area = None
        for coords in variants:
            # coords is a set of (x,y)
            maxx = max(x for x, y in coords)
            maxy = max(y for x, y in coords)
            wvar, hvar = maxx + 1, maxy + 1
            if area is None:
                area = len(coords)
            for oy in range(H - hvar + 1):
                for ox in range(W - wvar + 1):
                    mask = 0
                    off = oy * W + ox
                    for x, y in coords:
                        bit = off + y * W + x
                        mask |= 1 << bit
                    masks.add(mask)
        placements[idx] = list(masks)
        areas[idx] = area or 0
    return placements, areas, total_cells


def can_fit_with_placements(W, H, shape_counts, placements, areas, hole_check=True, time_limit=0.0, return_solution=False):
    """Bitmask-based backtracking with heuristics and memoization.

    shape_dict: {idx: [set of (x,y) coords variant]}
    shape_counts: list of counts per idx
    """
    total_cells = W * H

    # Quick area check
    required_area = sum(areas[i] * cnt for i, cnt in enumerate(shape_counts))
    if required_area > total_cells:
        return False

    # pre-check: any shape with placements empty while count>0 => fail
    for idx, cnt in enumerate(shape_counts):
        if cnt > 0 and not placements.get(idx):
            return False
    # pre-check: not enough distinct placements to fulfill count
    for idx, cnt in enumerate(shape_counts):
        if cnt > 0 and len(placements.get(idx, [])) < cnt:
            return False

    visited = set()  # memoize failed states

    # helper: count bits
    def bitcount(x):
        return x.bit_count()

    # connectivity check helper: if any empty area is smaller than smallest remaining piece, prune
    def has_small_hole(occ, min_area):
        if min_area <= 1:
            return False
        # create a set of empty coords to consider for BFS
        seen = 0
        fullmask = (1 << total_cells) - 1
        emptymask = (~occ) & fullmask
        if emptymask == 0:
            return min_area > 0
        # find any 0-bit quickly
        while emptymask:
            # get lowest set bit
            lb = emptymask & -emptymask
            # index of that bit
            start = (lb.bit_length() - 1)
            # BFS using bitmasks
            queue = [start]
            compmask = 0
            while queue:
                pos = queue.pop()
                if (compmask >> pos) & 1:
                    continue
                compmask |= 1 << pos
                x = pos % W
                y = pos // W
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < W and 0 <= ny < H:
                        npos = ny * W + nx
                        if ((emptymask >> npos) & 1) and not ((compmask >> npos) & 1):
                            queue.append(npos)
            # remove compmask from emptymask
            if bitcount(compmask) < min_area:
                return True
            emptymask &= ~compmask
        return False

    # choose next shape with minimal candidates given current occupancy
    def select_shape(occ, counts):
        best_idx = None
        best_count = None
        best_candidates = None
        for idx, cnt in enumerate(counts):
            if cnt == 0:
                continue
            # compute candidate placements not overlapping
            cand = [m for m in placements[idx] if (m & occ) == 0]
            cnum = len(cand)
            if cnum == 0:
                return idx, cand
            if best_count is None or cnum < best_count:
                best_count = cnum
                best_idx = idx
                best_candidates = cand
                if best_count <= 1:
                    break
        return best_idx, best_candidates

    # DFS
    tstart = time.perf_counter()

    solution_masks = []

    def dfs(occ, counts, remaining_area):
        if time_limit and (time.perf_counter() - tstart) > time_limit:
            raise TimeoutError()
        if all(c == 0 for c in counts):
            return True
        key = (occ, tuple(counts))
        if key in visited:
            return False
        # area check
        if total_cells - bitcount(occ) < remaining_area:
            visited.add(key)
            return False
        # choose shape
        idx, candidates = select_shape(occ, counts)
        if idx is None:
            # no shapes left
            return True
        if not candidates:
            visited.add(key)
            return False
        # optional connectivity pruning
        min_area = min((areas[i] for i, c in enumerate(counts) if c > 0), default=1)
        if hole_check and has_small_hole(occ, min_area):
            visited.add(key)
            return False
        for m in candidates:
            new_occ = occ | m
            counts[idx] -= 1
            solution_masks.append((idx, m))
            if dfs(new_occ, counts, remaining_area - areas[idx]):
                return True
            solution_masks.pop()
            counts[idx] += 1
        visited.add(key)
        return False

    start_occ = 0
    counts_copy = list(shape_counts)
    try:
        ok = dfs(start_occ, counts_copy, required_area)
        if return_solution:
            return ok, list(solution_masks)
        return ok
    except TimeoutError:
        if return_solution:
            return None, None
        return None


def build_exact_cover_matrix(W, H, shape_counts, placements):
    """Return (num_cols, rows) where rows is list of sets of column indices for exact cover.

    Columns: 0..W*H-1 -> cell positions.
    Next columns: for each shape idx create shape_counts[idx] instance columns.
    Rows: for each placement of shape idx (mask), create one row per instance column: cells covered + that instance col index.
    """
    cell_cols = W * H
    # compute instance columns offsets
    instance_col_start = cell_cols
    instance_cols = {}
    cur = instance_col_start
    for idx, cnt in enumerate(shape_counts):
        instance_cols[idx] = list(range(cur, cur + cnt))
        cur += cnt
    num_cols = cur
    rows = []
    # For each shape idx and each placement mask, create rows
    for idx, masks in placements.items():
        inst_cols = instance_cols.get(idx, [])
        if not inst_cols:
            continue
        for mask in masks:
            # compute cell columns from mask
            cols = set()
            bit = 0
            m = mask
            while m:
                lb = m & -m
                pos = lb.bit_length() - 1
                cols.add(pos)
                m &= m - 1
            # for each instance, create a row with those cells + instance col
            for ic in inst_cols:
                rows.append(frozenset(cols | {ic}))
    # Add filler rows to allow empty cells (each cell can be left empty)
    for c in range(cell_cols):
        rows.append(frozenset({c}))
    return num_cols, rows


class DLXNode:
    __slots__ = ('L', 'R', 'U', 'D', 'C', 'row_id')
    def __init__(self):
        self.L = self.R = self.U = self.D = self
        self.C = None
        self.row_id = None


class ColumnHeader(DLXNode):
    __slots__ = ('name', 'size', 'is_primary')
    def __init__(self, name, is_primary=True):
        super().__init__()
        self.name = name
        self.size = 0
        self.is_primary = is_primary


class DancingLinks:
    def __init__(self):
        self.header = ColumnHeader('header', is_primary=True)
        self.header.L = self.header.R = self.header
        self.columns = []  # ordered list of ColumnHeader
        self.col_map = {}

    def add_column(self, name, is_primary=True):
        col = ColumnHeader(name, is_primary=is_primary)
        col.L = self.header.L
        col.R = self.header
        self.header.L.R = col
        self.header.L = col
        self.columns.append(col)
        self.col_map[name] = col
        return col

    def add_row(self, row_id, cols):
        first = None
        last = None
        for c in cols:
            col = self.col_map[c]
            node = DLXNode()
            node.C = col
            node.row_id = row_id
            # insert into column (bottom)
            node.U = col.U
            node.D = col
            col.U.D = node
            col.U = node
            col.size += 1
            if first is None:
                first = node
                last = node
                node.L = node.R = node
            else:
                node.L = last
                node.R = first
                last.R = node
                first.L = node
                last = node

    def cover(self, col):
        col.R.L = col.L
        col.L.R = col.R
        i = col.D
        while i != col:
            j = i.R
            while j != i:
                j.D.U = j.U
                j.U.D = j.D
                j.C.size -= 1
                j = j.R
            i = i.D

    def uncover(self, col):
        i = col.U
        while i != col:
            j = i.L
            while j != i:
                j.C.size += 1
                j.D.U = j
                j.U.D = j
                j = j.L
            i = i.U
        col.R.L = col
        col.L.R = col

    def search(self, max_time=0.0):
        start = time.perf_counter()
        solution = []

        def choose_column():
            # choose primary column with minimal size
            best = None
            h = self.header.R
            while h != self.header:
                if getattr(h, 'is_primary', False):
                    if best is None or h.size < best.size:
                        best = h
                h = h.R
            return best

        def _search():
            if max_time and (time.perf_counter() - start) > max_time:
                raise TimeoutError()
            c = choose_column()
            if c is None:
                return True
            self.cover(c)
            r = c.D
            while r != c:
                solution.append(r.row_id)
                j = r.R
                while j != r:
                    self.cover(j.C)
                    j = j.R
                if _search():
                    return True
                j = r.L
                while j != r:
                    self.uncover(j.C)
                    j = j.L
                solution.pop()
                r = r.D
            self.uncover(c)
            return False

        try:
            res = _search()
            return res, list(solution)
        except TimeoutError:
            return None, None


def dlx_solve(num_cols, rows, time_limit=0.0):
    """Solve exact cover using Algorithm X with simple backtracking and set structure.
    Returns True if cover exists, False if not, None if timed out.
    """
    start = time.perf_counter()
    # map column -> set(rows idx)
    X = {c: set() for c in range(num_cols)}
    Y = {}
    for i, r in enumerate(rows):
        Y[i] = set(r)
        for c in r:
            X[c].add(i)

    solution = []

    def search():
        if time_limit and (time.perf_counter() - start) > time_limit:
            raise TimeoutError()
        if not X:
            return True
        # choose column with minimum size
        c = min(X, key=lambda c: len(X[c]))
        if not X[c]:
            return False
        rows_for_c = list(X[c])
        for r in rows_for_c:
            solution.append(r)
            cols_covered = []
            rows_removed = {}
            # cover columns in Y[r]
            for j in Y[r]:
                cols_covered.append(j)
                # remove j from X and also remove rows that have j from other columns
                rows_removed[j] = X.pop(j)
                for i2 in list(rows_removed[j]):
                    for j2 in Y[i2]:
                        if j2 != j and j2 in X:
                            X[j2].remove(i2)
            try:
                if search():
                    return True
            finally:
                # restore
                for j in reversed(cols_covered):
                    rows_set = rows_removed.pop(j)
                    X[j] = rows_set
                    for i2 in rows_set:
                        for j2 in Y[i2]:
                            if j2 != j and j2 in X:
                                X[j2].add(i2)
                solution.pop()
        return False

    try:
        res = search()
        return res
    except TimeoutError:
        return None


def can_fit_dlx(W, H, shape_counts, placements, time_limit=0.0, return_solution=False):
    # simple checks
    total_cells = W * H
    # compute area per shape as length of any placement (we can derive from mask bits)
    areas = {idx: (masks[0].bit_count() if masks else 0) for idx, masks in placements.items()}
    required_area = sum(areas.get(i, 0) * cnt for i, cnt in enumerate(shape_counts))
    if required_area > total_cells:
        return False
    # ensure each shape with count>0 has placements
    for idx, cnt in enumerate(shape_counts):
        if cnt > 0 and not placements.get(idx):
            return False
    # pre-check: not enough distinct placements to fulfill count
    for idx, cnt in enumerate(shape_counts):
        if cnt > 0 and len(placements.get(idx, [])) < cnt:
            return False
    # Build DLX with primary columns = instance columns, secondary = cell columns
    cell_cols = W * H
    dlx = DancingLinks()
    # add cell columns as secondary
    for c in range(cell_cols):
        dlx.add_column(c, is_primary=False)
    # instance columns
    instance_col_start = cell_cols
    cur = instance_col_start
    instance_cols = {}
    for idx, cnt in enumerate(shape_counts):
        instance_cols[idx] = list(range(cur, cur + cnt))
        for ic in instance_cols[idx]:
            dlx.add_column(ic, is_primary=True)
        cur += cnt

    # add rows: for each placement mask and each instance column, row = cell bits + instance column
    rowid = 0
    rowid_map = {}
    for idx, masks in placements.items():
        for mask in masks:
            cols = []
            m = mask
            while m:
                lb = m & -m
                pos = lb.bit_length() - 1
                cols.append(pos)
                m &= m - 1
            for ic in instance_cols.get(idx, []):
                dlx.add_row(rowid, cols + [ic])
                rowid_map[rowid] = (idx, mask)
                rowid += 1

    res, sol = dlx.search(max_time=time_limit)
    if return_solution:
        if res and sol:
            placed = [rowid_map[rid] for rid in sol]
            return res, placed
        return res, None
    return res


def validate_solution(W, H, shape_counts, placement_list):
    """Validate that placement_list is a valid set of placements: list of tuples (idx, mask)."""
    if placement_list is None:
        return False, "no solution"
    fullmask = 0
    counts = [0] * len(shape_counts)
    for idx, mask in placement_list:
        if fullmask & mask:
            return False, "overlap"
        fullmask |= mask
        counts[idx] += 1
    # check counts equal required
    for i, c in enumerate(shape_counts):
        if counts[i] != c:
            return False, f"counts mismatch for idx {i}: {counts[i]} != {c}"
    return True, "ok"


def can_fit(region_w, region_h, shape_counts, shape_dict):
    placements, areas, _ = compute_placements_for_size(shape_dict, region_w, region_h)
    return can_fit_with_placements(region_w, region_h, shape_counts, placements, areas)

def read_input_file(filename="input"):
    with open(filename, "r") as f:
        return f.read()

_WORKER_PLACEMENTS = None


def worker_init(placements_map):
    global _WORKER_PLACEMENTS
    _WORKER_PLACEMENTS = placements_map


def solve_region_tuple(args):
    W, H, counts, idx, hole_check, timeout, method, fallback = args
    placements, areas, _ = _WORKER_PLACEMENTS[(W, H)]
    if method == 'dlx':
        ok = can_fit_dlx(W, H, counts, placements, time_limit=timeout)
        if ok is None and fallback:
            ok = can_fit_with_placements(W, H, counts, placements, areas, hole_check=hole_check, time_limit=fallback)
    else:
        ok = can_fit_with_placements(W, H, counts, placements, areas, hole_check=hole_check, time_limit=timeout)
    return idx, ok


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--example", action="store_true", help="Run the example input")
    parser.add_argument("--input", default="input", help="Input filename")
    parser.add_argument("--workers", type=int, default=0, help="Number of worker processes (0=serial)")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of regions to process (0=all)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--hole-check", action="store_true", help="Enable small-hole connectivity pruning")
    parser.add_argument("--timeout", type=float, default=0.0, help="Per-region timeout seconds (0=no timeout)")
    parser.add_argument("--method", choices=["bitmask", "dlx"], default="bitmask", help="Solving method to use")
    parser.add_argument("--fallback", type=float, default=0.0, help="Fallback per-region timeout for bitmask solver when DLX times out (0=disabled)")
    parser.add_argument("--compare", action="store_true", help="Compare bitmask and dlx outputs for initial regions")
    parser.add_argument("--compare-n", type=int, default=10, help="Number of initial regions to compare when --compare is used")
    args = parser.parse_args(argv)

    if args.example:
        text = example_input
    else:
        text = read_input_file(args.input)
    shapes, regions = parse_input(text)
    # Convert shapes to coordinate sets
    shape_dict = {}
    for idx, grid in shapes.items():
        variants = []
        for var in shape_variants(grid):
            coords = {(x, y) for y, row in enumerate(var) for x, c in enumerate(row) if c == '#'}
            # normalize coordinates (translate to origin)
            if coords:
                minx = min(x for x, y in coords)
                miny = min(y for x, y in coords)
                coords = {(x - minx, y - miny) for x, y in coords}
            if coords not in variants:
                variants.append(coords)
        shape_dict[idx] = variants
    # Precompute placements for unique sizes
    unique_sizes = sorted({(w, h) for w, h, _ in regions})
    placements_by_wh = {}
    for W, H in unique_sizes:
        placements_by_wh[(W, H)] = compute_placements_for_size(shape_dict, W, H)

    # Optionally limit
    if args.limit > 0:
        regions = regions[: args.limit]

    # Build tasks
    tasks = []
    for idx, (w, h, counts) in enumerate(regions, start=1):
        tasks.append((w, h, counts, idx, args.hole_check, args.timeout, args.method, args.fallback))

    if args.compare:
        mismatches = []
        cmp_n = min(len(regions), args.compare_n)
        print(f"Comparing first {cmp_n} regions using both solvers (may be slow)")
        for i in range(cmp_n):
            w, h, counts = regions[i]
            placements, areas, _ = placements_by_wh[(w, h)]
            bm = can_fit_with_placements(w, h, counts, placements, areas, hole_check=args.hole_check, time_limit=args.timeout)
            dlx_res = can_fit_dlx(w, h, counts, placements, time_limit=args.timeout)
            print(f"Region {i+1}: bitmask={bm} dlx={dlx_res}")
            # if mismatch, try to get concrete solutions and validate
            if bm != dlx_res:
                bm_ok, bm_sol = can_fit_with_placements(w, h, counts, placements, areas, hole_check=args.hole_check, time_limit=args.timeout, return_solution=True)
                dlx_ok, dlx_sol = can_fit_dlx(w, h, counts, placements, time_limit=args.timeout, return_solution=True)
                bm_val = validate_solution(w, h, counts, bm_sol)[1] if bm_sol is not None else "bm no sol"
                dlx_val = validate_solution(w, h, counts, dlx_sol)[1] if dlx_sol is not None else "dlx no sol"
                print(f"  bitmask solution check: {bm_ok} {bm_val}")
                print(f"  dlx solution check: {dlx_ok} {dlx_val}")
                mismatches.append((i+1, w, h, counts, bm, dlx_res, bm_ok, bm_val, dlx_ok, dlx_val))
        if mismatches:
            print("MISMATCHES: ")
            for m in mismatches:
                print(m)
        else:
            print("No mismatches for compared regions.")

    start_time = time.perf_counter()
    count = 0
    if args.workers and args.workers > 0:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=worker_init, initargs=(placements_by_wh,)) as ex:
            futures = {ex.submit(solve_region_tuple, t): t[3] for t in tasks}
            for future in as_completed(futures):
                idx, ok = future.result()
                if ok:
                    count += 1
                if args.verbose:
                    print(f"Region {idx}: {'OK' if ok else 'NO'}")
    else:
        for w, h, counts, idx, hole_check, timeout, method, fallback in tasks:
            placements, areas, _ = placements_by_wh[(w, h)]
            if method == 'dlx':
                if args.verbose:
                    num_cols, rows = build_exact_cover_matrix(w, h, counts, placements)
                    print(f"Region {idx}: DLX num_cols={num_cols} rows={len(rows)}")
                ok = can_fit_dlx(w, h, counts, placements, time_limit=timeout)
                if ok is None and fallback:
                    # DLX timed out; try bitmask fallback
                    if args.verbose:
                        print(f"Region {idx}: DLX timed out, trying bitmask fallback {fallback}s")
                    ok = can_fit_with_placements(w, h, counts, placements, areas, hole_check=hole_check, time_limit=fallback)
            else:
                ok = can_fit_with_placements(w, h, counts, placements, areas, hole_check=hole_check, time_limit=timeout)
            if ok:
                count += 1
            if args.verbose:
                print(f"Region {idx}: {'OK' if ok else 'NO'}")
    end_time = time.perf_counter()
    print(count)
    print(f"Elapsed: {end_time - start_time:.3f}s")


if __name__ == "__main__":
    main()

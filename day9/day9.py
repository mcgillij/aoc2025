part1 = """
--- Day 9: Movie Theater ---

You slide down the firepole in the corner of the playground and land in the North Pole base movie theater!

The movie theater has a big tile floor with an interesting pattern. Elves here are redecorating the theater by switching out some of the square tiles in the big grid they form. Some of the tiles are red; the Elves would like to find the largest rectangle that uses red tiles for two of its opposite corners. They even have a list of where the red tiles are located in the grid (your puzzle input).

For example:

7,1
11,1
11,7
9,7
9,5
2,5
2,3
7,3

Showing red tiles as # and other tiles as ., the above arrangement of red tiles would look like this:

..............
.......#...#..
..............
..#....#......
..............
..#......#....
..............
.........#.#..
..............

You can choose any two red tiles as the opposite corners of your rectangle; your goal is to find the largest rectangle possible.

For example, you could make a rectangle (shown as O) with an area of 24 between 2,5 and 9,7:

..............
.......#...#..
..............
..#....#......
..............
..OOOOOOOO....
..OOOOOOOO....
..OOOOOOOO.#..
..............

Or, you could make a rectangle with area 35 between 7,1 and 11,7:

..............
.......OOOOO..
.......OOOOO..
..#....OOOOO..
.......OOOOO..
..#....OOOOO..
.......OOOOO..
.......OOOOO..
..............

You could even make a thin rectangle with an area of only 6 between 7,3 and 2,3:

..............
.......#...#..
..............
..OOOOOO......
..............
..#......#....
..............
.........#.#..
..............

Ultimately, the largest rectangle you can make in this example has area 50. One way to do this is between 2,5 and 11,1:

..............
..OOOOOOOOOO..
..OOOOOOOOOO..
..OOOOOOOOOO..
..OOOOOOOOOO..
..OOOOOOOOOO..
..............
.........#.#..
..............
"""

part2 = """
The Elves just remembered: they can only switch out tiles that are red or green. So, your rectangle can only include red or green tiles.

In your list, every red tile is connected to the red tile before and after it by a straight line of green tiles. The list wraps, so the first red tile is also connected to the last red tile. Tiles that are adjacent in your list will always be on either the same row or the same column.

Using the same example as before, the tiles marked X would be green:

..............
.......#XXX#..
.......X...X..
..#XXXX#...X..
..X........X..
..#XXXXXX#.X..
.........X.X..
.........#X#..
..............

In addition, all of the tiles inside this loop of red and green tiles are also green. So, in this example, these are the green tiles:

..............
.......#XXX#..
.......XXXXX..
..#XXXX#XXXX..
..XXXXXXXXXX..
..#XXXXXX#XX..
.........XXX..
.........#X#..
..............

The remaining tiles are never red nor green.

The rectangle you choose still must have red tiles in opposite corners, but any other tiles it includes must now be red or green. This significantly limits your options.

For example, you could make a rectangle out of red and green tiles with an area of 15 between 7,3 and 11,1:

..............
.......OOOOO..
.......OOOOO..
..#XXXXOOOOO..
..XXXXXXXXXX..
..#XXXXXX#XX..
.........XXX..
.........#X#..
..............

Or, you could make a thin rectangle with an area of 3 between 9,7 and 9,5:

..............
.......#XXX#..
.......XXXXX..
..#XXXX#XXXX..
..XXXXXXXXXX..
..#XXXXXXOXX..
.........OXX..
.........OX#..
..............

The largest rectangle you can make in this example using only red and green tiles has area 24. One way to do this is between 9,5 and 2,3:

..............
.......#XXX#..
.......XXXXX..
..OOOOOOOOXX..
..OOOOOOOOXX..
..OOOOOOOOXX..
.........XXX..
.........#X#..
..............

Using two red tiles as opposite corners, what is the largest area of any rectangle you can make using only red and green tiles?
"""

import re
import math
from bisect import bisect_right
import time

example_input = """7,1
11,1
11,7
9,7
9,5
2,5
2,3
7,3"""

def parse_red_tiles(puzzle):
    # Extract lines with coordinates (format: x,y)
    coords = re.findall(r'(\d+),(\d+)', puzzle)
    return [(int(x), int(y)) for x, y in coords]

def largest_rectangle_area(red_tiles):
    max_area = 0
    n = len(red_tiles)
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = red_tiles[i]
            x2, y2 = red_tiles[j]
            area = (abs(x1 - x2) + 1) * (abs(y1 - y2) + 1)
            # Rectangle area: abs(width) * abs(height)
#            area = abs(x1 - x2) * abs(y1 - y2)
            if area > max_area:
                max_area = area
    return max_area

def precompute_column_row_intervals(red_tiles, boundary_green=None):
    """
    For each compressed x-column (based on unique red xs), compute the list of
    integer row ranges that have their tile centers inside the polygon. This allows
    quick checks whether any contiguous integer y-range is entirely contained in
    the polygon for that column.
    """
    from bisect import bisect_right
    xs = sorted(set(x for x, _ in red_tiles))
    ys = sorted(set(y for _, y in red_tiles))
    xs2 = xs + [max(xs) + 1]
    ys2 = ys + [max(ys) + 1]

    # For each column index compute intervals
    col_intervals = []
    # Precompute boundary map (col index -> set of y rows) if provided
    boundary_map = {i: set() for i in range(len(xs))}
    if boundary_green is not None:
        for (bx, by) in boundary_green:
            # map bx to column index
            col_idx = bisect_right(xs, bx) - 1
            if 0 <= col_idx < len(xs):
                boundary_map[col_idx].add(by)

    for i in range(len(xs)):
        # Pick a point just to the right of integer x coordinate to avoid vertical-edge degeneracy
        x_rep = xs2[i] + 1e-9
        crosses = []
        n = len(red_tiles)
        for k in range(n):
            x1, y1 = red_tiles[k]
            x2, y2 = red_tiles[(k + 1) % n]
            # Check if edge crosses vertical line at x_rep
            if (x1 > x_rep) != (x2 > x_rep):
                # Interpolate y
                if x2 != x1:
                    y_cross = y1 + (y2 - y1) * (x_rep - x1) / (x2 - x1)
                    crosses.append(y_cross)
        crosses.sort()
        # Pair up crosses to intervals
        intervals = []
        for a, b in zip(crosses[::2], crosses[1::2]):
            # compute integer rows y such that left-bottom point (x, y) lies strictly inside (a, b)
            # i.e., y with a < y < b -> integer rows y in [floor(a)+1, ceil(b)-1]
            y_min_row = int(math.floor(a)) + 1
            y_max_row = int(math.ceil(b)) - 1
            if y_max_row >= y_min_row:
                intervals.append((y_min_row, y_max_row))
        # Add boundary single-row tiles from boundary_map
        if boundary_green is not None and boundary_map[i]:
            for by in boundary_map[i]:
                intervals.append((by, by))
        # merge intervals if needed (sort first by start)
        intervals.sort(key=lambda x: x[0])
        merged = []
        for s, e in intervals:
            if not merged or s > merged[-1][1] + 1:
                merged.append([s, e])
            else:
                merged[-1][1] = max(merged[-1][1], e)
        col_intervals.append([(s, e) for s, e in merged])

    return xs2, ys2, col_intervals


def largest_rectangle_area_red_green_intervals(red_tiles):
    """
    Use per-column integer row coverage intervals to check candidate rectangles.
    This avoids building full masks and does exact integer-tile membership checks
    across candidate rectangles using per-column intervals.
    """
    # Build boundary green tiles (edges between consecutive red tiles)
    n = len(red_tiles)
    boundary_green = set()
    for k in range(n):
        x1, y1 = red_tiles[k]
        x2, y2 = red_tiles[(k + 1) % n]
        if x1 == x2:
            for y in range(min(y1, y2), max(y1, y2) + 1):
                boundary_green.add((x1, y))
        elif y1 == y2:
            for x in range(min(x1, x2), max(x1, x2) + 1):
                boundary_green.add((x, y1))
    xs2, ys2, col_intervals = precompute_column_row_intervals(red_tiles, boundary_green)
    xs = xs2[:-1]
    min_x = xs[0]
    min_y = sorted(set(y for _, y in red_tiles))[0]

    def ix(x):
        return bisect_right(xs, x) - 1

    def check_col_cover(i, min_ry, max_ry):
        # Returns true if column i covers all integer rows from min_ry..max_ry inclusive
        intervals = col_intervals[i]
        # Binary search interval that might cover min_ry
        lo = 0
        hi = len(intervals) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            s, e = intervals[mid]
            if e < min_ry:
                lo = mid + 1
            elif s > min_ry:
                hi = mid - 1
            else:
                # Found interval that starts <= min_ry <= end
                cur_end = e
                idx = mid + 1
                while idx < len(intervals) and intervals[idx][0] <= cur_end + 1:
                    cur_end = max(cur_end, intervals[idx][1])
                    idx += 1
                return cur_end >= max_ry
        return False

    # Enumerate candidate rectangles
    reds = list(red_tiles)
    max_area = 0
    best_rect = None
    n = len(reds)
    for i in range(n):
        x1, y1 = reds[i]
        for j in range(i + 1, n):
            x2, y2 = reds[j]
            if x1 == x2 or y1 == y2:
                continue
            min_rx, max_rx = min(x1, x2), max(x1, x2)
            min_ry, max_ry = min(y1, y2), max(y1, y2)
            area = (max_rx - min_rx + 1) * (max_ry - min_ry + 1)
            if area <= max_area:
                continue
            min_ix = ix(min_rx)
            max_ix = ix(max_rx)
            ok = True
            for ci in range(min_ix, max_ix + 1):
                if not check_col_cover(ci, min_ry, max_ry):
                    ok = False
                    break
            if ok:
                max_area = area
                best_rect = (x1, y1, x2, y2)
    return max_area, best_rect

def largest_rectangle_area_red_green_set(red_tiles, all_green):
    max_area = 0
    red_list = list(red_tiles)
    for i in range(len(red_list)):
        x1, y1 = red_list[i]
        for j in range(i + 1, len(red_list)):
            x2, y2 = red_list[j]
            if x1 == x2 or y1 == y2:
                continue
            min_rx, max_rx = min(x1, x2), max(x1, x2)
            min_ry, max_ry = min(y1, y2), max(y1, y2)
            area = (max_rx - min_rx + 1) * (max_ry - min_ry + 1)
            if area <= max_area:
                continue
            # Check perimeter first
            valid = True
            for x in range(min_rx, max_rx + 1):
                if (x, min_ry) not in all_green or (x, max_ry) not in all_green:
                    valid = False
                    break
            if not valid:
                continue
            for y in range(min_ry, max_ry + 1):
                if (min_rx, y) not in all_green or (max_rx, y) not in all_green:
                    valid = False
                    break
            if not valid:
                continue
            # Check interior only if perimeter is valid
            for x in range(min_rx + 1, max_rx):
                for y in range(min_ry + 1, max_ry):
                    if (x, y) not in all_green:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                max_area = area
    return max_area

def read_input_file(filename="input"):
    with open(filename, "r") as f:
        return f.read()

if __name__ == "__main__":
    red_tiles_example = parse_red_tiles(example_input)
    print("example (part1):", largest_rectangle_area(red_tiles_example))
    # Example part2 result (intervals-based):
    ex_area, ex_rect = largest_rectangle_area_red_green_intervals(red_tiles_example)
    print("example (part2):", ex_area, "rect:", ex_rect)
    red_tiles_part1 = parse_red_tiles(read_input_file())
    print("part1:", largest_rectangle_area(red_tiles_part1))

    # Fast part2: compute boundary green tiles (edges) only, not interior — interval method handles interior.
    xs = [x for x, _ in red_tiles_part1]
    ys = [y for _, y in red_tiles_part1]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    # Build green boundary tiles (lines between consecutive red tiles)
    n = len(red_tiles_part1)
    boundary_green = set()
    for i in range(n):
        x1, y1 = red_tiles_part1[i]
        x2, y2 = red_tiles_part1[(i + 1) % n]
        if x1 == x2:
            for y in range(min(y1, y2), max(y1, y2) + 1):
                boundary_green.add((x1, y))
        elif y1 == y2:
            for x in range(min(x1, x2), max(x1, x2) + 1):
                boundary_green.add((x, y1))
    boundary_green = set()
    n = len(red_tiles_part1)
    for i in range(n):
        x1, y1 = red_tiles_part1[i]
        x2, y2 = red_tiles_part1[(i + 1) % n]
        if x1 == x2:
            for y in range(min(y1, y2), max(y1, y2) + 1):
                boundary_green.add((x1, y))
        elif y1 == y2:
            for x in range(min(x1, x2), max(x1, x2) + 1):
                boundary_green.add((x, y1))

    print(f"boundary tiles: {len(boundary_green)}")
    red_xs = [x for x, y in red_tiles_part1]
    red_ys = [y for x, y in red_tiles_part1]
    print(f"red_tiles size: {len(red_tiles_part1)}")
    print(f"red_tiles x range: {min(red_xs)} - {max(red_xs)}")
    print(f"red_tiles y range: {min(red_ys)} - {max(red_ys)}")
    print("Sample red_tiles:", list(red_tiles_part1)[:10])
    # Optionally, if boundary is small we can run the set-check version for validation.
    if len(boundary_green) < 300000:
        # Build a small all_green set only from the boundary + red to allow set-check for small polygons
        all_green_small = boundary_green | set(red_tiles_part1)
        print("part2 (set check):", largest_rectangle_area_red_green_set(red_tiles_part1, all_green_small))
    else:
        print("Skipping slow set-check (boundary too large)")
    # Try the interval-based approach
    t0 = time.time()
    inter_area, best_rect = largest_rectangle_area_red_green_intervals(red_tiles_part1)
    t1 = time.time()
    print(f"part2 (intervals col-check) final: {inter_area} (time {t1-t0:.3f}s)")
    print(f"best rectangle: {best_rect}")

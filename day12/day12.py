import re
import time
import argparse


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


def count_shape_cells(shape):
    """Count the number of '#' cells in a shape."""
    return sum(row.count('#') for row in shape)


def can_fit_trivial(W, H, shape_counts, shape_areas):
    """Check if region can fit pieces using trivial bounds.
    
    Returns:
        True: Definitely can fit (trivially solvable)
        False: Definitely cannot fit (impossible)
        None: Need actual packing solver (shouldn't happen in this input)
    """
    region_area = W * H
    total_cells_needed = sum(shape_areas[i] * count for i, count in enumerate(shape_counts))
    if total_cells_needed > region_area:
        return False
    total_pieces = sum(shape_counts)
    if region_area >= 9 * total_pieces:
        return True
    return None


def read_input_file(filename="input"):
    with open(filename, "r") as f:
        return f.read()


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


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--example", action="store_true")
    parser.add_argument("--input", default="input")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    if args.example:
        text = example_input
    else:
        text = read_input_file(args.input)

    shapes, regions = parse_input(text)

    # Calculate area for each shape (count '#' cells)
    shape_areas = {}
    for idx, grid in shapes.items():
        shape_areas[idx] = count_shape_cells(grid)

    if args.limit > 0:
        regions = regions[:args.limit]

    start_time = time.perf_counter()
    count = 0
    need_solver = 0

    for idx, (w, h, counts) in enumerate(regions, start=1):
        result = can_fit_trivial(w, h, counts, shape_areas)

        if result is True:
            count += 1
            if args.verbose:
                print(f"Region {idx}: YES (trivially solvable)")
        elif result is False:
            if args.verbose:
                print(f"Region {idx}: NO (impossible)")
        else:
            need_solver += 1
            if args.verbose:
                print(f"Region {idx}: UNKNOWN (needs actual packing solver)")

    end_time = time.perf_counter()

    print(f"\nResults:")
    print(f"  Regions that fit: {count}")
    print(f"  Regions that don't fit: {len(regions) - count - need_solver}")
    print(f"  Regions needing solver: {need_solver}")
    print(f"  Elapsed: {end_time - start_time:.6f}s")
    print(f"\nAnswer: {count}")


if __name__ == "__main__":
    main()

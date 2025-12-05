part1 = """
--- Day 5: Cafeteria ---

As the forklifts break through the wall, the Elves are delighted to discover that there was a cafeteria on the other side after all.

You can hear a commotion coming from the kitchen. "At this rate, we won't have any time left to put the wreaths up in the dining hall!" Resolute in your quest, you investigate.

"If only we hadn't switched to the new inventory management system right before Christmas!" another Elf exclaims. You ask what's going on.

The Elves in the kitchen explain the situation: because of their complicated new inventory management system, they can't figure out which of their ingredients are fresh and which are spoiled. When you ask how it works, they give you a copy of their database (your puzzle input).

The database operates on ingredient IDs. It consists of a list of fresh ingredient ID ranges, a blank line, and a list of available ingredient IDs. For example:

3-5
10-14
16-20
12-18

1
5
8
11
17
32

The fresh ID ranges are inclusive: the range 3-5 means that ingredient IDs 3, 4, and 5 are all fresh. The ranges can also overlap; an ingredient ID is fresh if it is in any range.

The Elves are trying to determine which of the available ingredient IDs are fresh. In this example, this is done as follows:

    Ingredient ID 1 is spoiled because it does not fall into any range.
    Ingredient ID 5 is fresh because it falls into range 3-5.
    Ingredient ID 8 is spoiled.
    Ingredient ID 11 is fresh because it falls into range 10-14.
    Ingredient ID 17 is fresh because it falls into range 16-20 as well as range 12-18.
    Ingredient ID 32 is spoiled.

So, in this example, 3 of the available ingredient IDs are fresh.

Process the database file from the new inventory management system. How many of the available ingredient IDs are fresh?
"""

part2 = """--- Part Two ---

The Elves start bringing their spoiled inventory to the trash chute at the back of the kitchen.

So that they can stop bugging you when they get new inventory, the Elves would like to know all of the IDs that the fresh ingredient ID ranges consider to be fresh. An ingredient ID is still considered fresh if it is in any range.

Now, the second section of the database (the available ingredient IDs) is irrelevant. Here are the fresh ingredient ID ranges from the above example:

3-5
10-14
16-20
12-18

The ingredient IDs that these ranges consider to be fresh are 3, 4, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, and 20. So, in this example, the fresh ingredient ID ranges consider a total of 14 ingredient IDs to be fresh.

Process the database file again. How many ingredient IDs are considered to be fresh according to the fresh ingredient ID ranges?
"""

def count_fresh_ingredients(puzzle_input: str) -> int:
    # Split input into lines and remove empty lines
    lines = [line.strip() for line in puzzle_input.strip().splitlines() if line.strip()]
    # Find the index where ranges end and IDs begin
    split_idx = 0
    for i, line in enumerate(lines):
        if '-' not in line:
            split_idx = i
            break
    # Parse ranges
    ranges = []
    for line in lines[:split_idx]:
        start, end = map(int, line.split('-'))
        ranges.append((start, end))
    # Parse available IDs
    ids = [int(line) for line in lines[split_idx:]]
    # Check freshness
    fresh_count = 0
    for id_ in ids:
        if any(start <= id_ <= end for start, end in ranges):
            fresh_count += 1
    return fresh_count

def count_total_fresh_ids(puzzle_input: str) -> int:
    # Split input into lines and remove empty lines
    lines = [line.strip() for line in puzzle_input.strip().splitlines() if line.strip()]
    # Only consider lines with ranges (contain '-')
    ranges = []
    for line in lines:
        if '-' in line:
            start, end = map(int, line.split('-'))
            ranges.append((start, end))
        else:
            break  # Stop at first non-range line

    # Merge overlapping ranges
    ranges.sort()
    merged = []
    for start, end in ranges:
        if not merged or merged[-1][1] < start - 1:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    # Count total unique IDs
    return sum(end - start + 1 for start, end in merged)

def read_input_file(filename: str = "input") -> str:
    with open(filename, "r") as f:
        return f.read()

# Example usage:
if __name__ == "__main__":
    example_input = """3-5
10-14
16-20
12-18

1
5
8
11
17
32
"""
    print("example: ", count_fresh_ingredients(example_input))  # Output: 3
    print("example2: ", count_total_fresh_ids(example_input))  # Output: 14
    part1_input = read_input_file()
    print("part1: ", count_fresh_ingredients(part1_input))
    print("part1: ", count_total_fresh_ids(part1_input))


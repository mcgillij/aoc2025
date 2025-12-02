day2_problem = """
As it turns out, one of the younger Elves was playing on a gift shop computer and managed to add a whole bunch of invalid product IDs to their gift shop database! Surely, it would be no trouble for you to identify the invalid product IDs for them, right?

They've even checked most of the product ID ranges already; they only have a few product ID ranges (your puzzle input) that you'll need to check. For example:

11-22,95-115,998-1012,1188511880-1188511890,222220-222224,
1698522-1698528,446443-446449,38593856-38593862,565653-565659,
824824821-824824827,2121212118-2121212124

(The ID ranges are wrapped here for legibility; in your input, they appear on a single long line.)

The ranges are separated by commas (,); each range gives its first ID and last ID separated by a dash (-).

Since the young Elf was just doing silly patterns, you can find the invalid IDs by looking for any ID which is made only of some sequence of digits repeated twice. So, 55 (5 twice), 6464 (64 twice), and 123123 (123 twice) would all be invalid IDs.

None of the numbers have leading zeroes; 0101 isn't an ID at all. (101 is a valid ID that you would ignore.)

Your job is to find all of the invalid IDs that appear in the given ranges. In the above example:

    11-22 has two invalid IDs, 11 and 22.
    95-115 has one invalid ID, 99.
    998-1012 has one invalid ID, 1010.
    1188511880-1188511890 has one invalid ID, 1188511885.
    222220-222224 has one invalid ID, 222222.
    1698522-1698528 contains no invalid IDs.
    446443-446449 has one invalid ID, 446446.
    38593856-38593862 has one invalid ID, 38593859.
    The rest of the ranges contain no invalid IDs.

Adding up all the invalid IDs in this example produces 1227775554.
"""

part2_problem = """
--- Part Two ---

The clerk quickly discovers that there are still invalid IDs in the ranges in your list. Maybe the young Elf was doing other silly patterns as well?

Now, an ID is invalid if it is made only of some sequence of digits repeated at least twice. So, 12341234 (1234 two times), 123123123 (123 three times), 1212121212 (12 five times), and 1111111 (1 seven times) are all invalid IDs.

From the same example as before:

    11-22 still has two invalid IDs, 11 and 22.
    95-115 now has two invalid IDs, 99 and 111.
    998-1012 now has two invalid IDs, 999 and 1010.
    1188511880-1188511890 still has one invalid ID, 1188511885.
    222220-222224 still has one invalid ID, 222222.
    1698522-1698528 still contains no invalid IDs.
    446443-446449 still has one invalid ID, 446446.
    38593856-38593862 still has one invalid ID, 38593859.
    565653-565659 now has one invalid ID, 565656.
    824824821-824824827 now has one invalid ID, 824824824.
    2121212118-2121212124 now has one invalid ID, 2121212121.

Adding up all the invalid IDs in this example produces 4174379265.

What do you get if you add up all of the invalid IDs using these new rules?
"""

def is_invalid_id_part1(n: int) -> bool:
    s = str(n)
    l = len(s)
    if l % 2 != 0:
        return False
    half = l // 2
    return s[:half] == s[half:]

def is_invalid_id(n: int) -> bool:
    s = str(n)
    l = len(s)
    for sub_len in range(1, l // 2 + 1):
        if l % sub_len == 0:
            if s == s[:sub_len] * (l // sub_len):
                return True
    return False

def find_invalid_ids(ranges: str) -> list[int]:
    ids = []
    for part in ranges.strip().split(','):
        if not part:
            continue
        start, end = map(int, part.split('-'))
        for n in range(start, end + 1):
            if is_invalid_id(n):
                ids.append(n)
    return ids

def sum_invalid_ids(ranges: str) -> int:
    return sum(find_invalid_ids(ranges))

def read_input(path: str = "input") -> str:
    with open(path) as f:
        return f.read().strip()


if __name__ == "__main__":
    example_ranges = (
        "11-22,95-115,998-1012,1188511880-1188511890,222220-222224,"
        "1698522-1698528,446443-446449,38593856-38593862,565653-565659,"
        "824824821-824824827,2121212118-2121212124"
    )
    invalid_ids = find_invalid_ids(example_ranges)
    print("Invalid IDs:", invalid_ids)
    print("Sum of invalid IDs:", sum(invalid_ids))
    full_input = read_input("input")
    full_invalid_ids = find_invalid_ids(full_input)
    print("Part1: Total Invalid IDs:", sum(full_invalid_ids))

    full_invalid_ids = find_invalid_ids(full_input)
    print("Part2: Total Invalid IDs:", sum(full_invalid_ids))

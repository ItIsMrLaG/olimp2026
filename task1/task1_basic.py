import sys
import struct


def ipv6_to_bytes(s):
    """Parse any valid IPv6 string into 16-byte big-endian canonical form."""
    s = s.strip().lower()
    if "::" in s:
        left, right = s.split("::", 1)
        lg = left.split(":") if left else []
        rg = right.split(":") if right else []
        groups = lg + ["0"] * (8 - len(lg) - len(rg)) + rg
    else:
        groups = s.split(":")
    return struct.pack(">8H", *(int(g, 16) for g in groups))


def task1(input_path, output_path):
    """Basic solution: store all addresses in a set in memory."""
    seen = set()
    with open(input_path, "r") as f:
        for line in f:
            seen.add(ipv6_to_bytes(line))
    with open(output_path, "w") as out:
        out.write(str(len(seen)) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: {} <input_file> <output_file>".format(sys.argv[0]))
        sys.exit(1)
    task1(sys.argv[1], sys.argv[2])


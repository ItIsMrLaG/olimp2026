import gc
import heapq
import os
import struct
import sys
import tempfile
from multiprocessing import Pool, cpu_count

# 200MB headroom from 1GB limit for OS, merge phase, Python runtime
MEMORY_BUDGET = 800 * 1024 * 1024
# 16B data + 33B PyObject header + 8B list pointer
BYTES_PER_RECORD = 57
# Below this chunk size merge degrades due to too many temp files
MIN_CHUNK = 500_000
# Records per buffered read from temp file during merge
READ_BUF = 4096


# --- IPv6 parsing ---


def _ipv6_to_bytes(s: str) -> bytes:
    """Parse any valid IPv6 string into 16-byte big-endian representation."""
    s = s.strip().lower()

    if "::" in s:
        left, right = s.split("::", 1)
        lg = left.split(":") if left else []
        rg = right.split(":") if right else []
        groups = lg + ["0"] * (8 - len(lg) - len(rg)) + rg
    else:
        groups = s.split(":")

    return struct.pack(">8H", *(int(g, 16) for g in groups))


# --- Chunk I/O ---


def _flush_chunk(chunk: list[bytes], temp_dir: str) -> str:
    """Sort chunk in-place, write as raw binary (16 bytes per record)."""
    chunk.sort()
    fd, path = tempfile.mkstemp(dir=temp_dir, suffix=".bin")

    with os.fdopen(fd, "wb", buffering=1 << 20) as f:
        for record in chunk:
            f.write(record)

    return path


def _sorted_records(fh, buf_size: int):
    """Yield 16-byte records from binary temp file with buffered reads."""
    while True:
        data = fh.read(16 * buf_size)
        if not data:
            break
        for i in range(0, len(data), 16):
            yield data[i : i + 16]


# --- Parallel segmentation ---


def _get_segments(input_path: str, num_segments: int) -> list[tuple[int, int]]:
    """Split file into byte-range segments aligned to newline boundaries."""
    file_size = os.path.getsize(input_path)
    if file_size == 0:
        return []

    seg_size = file_size // num_segments
    boundaries = [0]

    with open(input_path, "rb") as f:
        for i in range(1, num_segments):
            f.seek(min(i * seg_size, file_size))
            f.readline()
            pos = f.tell()
            if pos < file_size and pos != boundaries[-1]:
                boundaries.append(pos)

    boundaries.append(file_size)
    return [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]


def _worker(args: tuple) -> list[str]:
    """Read a file segment, parse IPv6 lines, flush sorted chunks to temp files."""
    input_path, start, end, temp_dir, chunk_size = args
    temp_paths: list[str] = []
    chunk: list[bytes] = []

    with open(input_path, "rb") as f:
        f.seek(start)
        pos = start
        while pos < end:
            raw = f.readline()
            if not raw:
                break
            pos += len(raw)
            line = raw.decode("ascii", errors="ignore").strip()
            if line:
                chunk.append(_ipv6_to_bytes(line))
                if len(chunk) >= chunk_size:
                    temp_paths.append(_flush_chunk(chunk, temp_dir))
                    chunk.clear()
                    gc.collect()

    if chunk:
        temp_paths.append(_flush_chunk(chunk, temp_dir))

    return temp_paths


# --- Merge + count ---


def _merge_and_count(temp_files: list[str]) -> int:
    """K-way merge via min-heap, count unique records."""
    buf_per_file = max(256, min(READ_BUF, 5_000_000 // max(len(temp_files), 1)))
    file_handles = [open(p, "rb") for p in temp_files]
    iterators = [_sorted_records(fh, buf_per_file) for fh in file_handles]

    unique_count = 0
    prev = None

    for val in heapq.merge(*iterators):
        if val != prev:
            unique_count += 1
            prev = val

    for fh in file_handles:
        fh.close()

    return unique_count


# --- Cleanup ---


def _cleanup(temp_files: list[str], temp_dir: str) -> None:
    """Remove all temp files and directory."""
    for p in temp_files:
        os.remove(p)
    os.rmdir(temp_dir)


# --- Public API ---


def task1(input_path: str, output_path: str) -> None:
    """Count unique IPv6 addresses in input file using parallel external sort."""
    temp_dir = tempfile.mkdtemp()

    # Determine worker count from CPU and memory constraints
    max_by_cpu = max(1, cpu_count() - 1)
    max_by_mem = max(1, MEMORY_BUDGET // (MIN_CHUNK * BYTES_PER_RECORD))
    num_workers = min(max_by_cpu, max_by_mem)

    # Distribute memory budget evenly across workers
    chunk_per_worker = max(MIN_CHUNK, MEMORY_BUDGET // (num_workers * BYTES_PER_RECORD))

    segments = _get_segments(input_path, num_workers)
    if not segments:
        with open(output_path, "w") as out:
            out.write("0\n")
        os.rmdir(temp_dir)
        return

    # Phase 1: parallel parse + sort into binary temp files
    worker_args = [
        (input_path, start, end, temp_dir, chunk_per_worker)
        for start, end in segments
    ]

    with Pool(num_workers) as pool:
        results = pool.map(_worker, worker_args)

    temp_files = [p for paths in results for p in paths]
    if not temp_files:
        with open(output_path, "w") as out:
            out.write("0\n")
        os.rmdir(temp_dir)
        return

    # Phase 2: k-way merge, count unique addresses
    unique_count = _merge_and_count(temp_files)
    _cleanup(temp_files, temp_dir)

    with open(output_path, "w") as out:
        out.write(f"{unique_count}\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input_file> <output_file>")
        sys.exit(1)
    task1(sys.argv[1], sys.argv[2])

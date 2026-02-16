import os
import sys
import struct
import heapq
import tempfile
import gc
from multiprocessing import Pool, cpu_count

# Reserve 200MB headroom from 1GB limit for OS, merge phase, Python runtime
MEMORY_BUDGET = 800 * 1024 * 1024
# Per-record cost: 16B data + 33B PyObject header + 8B list pointer
BYTES_PER_RECORD = 57
# Below this chunk size merge degrades due to too many temp files
MIN_CHUNK = 500_000
# Records per buffered read from temp file during merge
READ_BUF = 4096


def ipv6_to_bytes(s):
    """Parse any valid IPv6 form (::, leading zeros, mixed case) into
    16-byte big-endian. Lexicographic order of result == numeric order."""
    s = s.strip().lower()
    if "::" in s:
        left, right = s.split("::", 1)
        lg = left.split(":") if left else []
        rg = right.split(":") if right else []
        # Fill missing groups with zeros to always get exactly 8
        groups = lg + ["0"] * (8 - len(lg) - len(rg)) + rg
    else:
        groups = s.split(":")
    return struct.pack(">8H", *(int(g, 16) for g in groups))


def flush_chunk(chunk, temp_dir):
    """Sort chunk in-place, write as raw binary (16 bytes per record).
    Per-record write with 1MB buffer avoids b"".join peak allocation."""
    chunk.sort()
    fd, path = tempfile.mkstemp(dir=temp_dir, suffix=".bin")
    with os.fdopen(fd, "wb", buffering=1 << 20) as f:
        for record in chunk:
            f.write(record)
    return path


def sorted_records(fh, buf_size):
    """Yield 16-byte records from binary temp file with buffered reads."""
    while True:
        data = fh.read(16 * buf_size)
        if not data:
            break
        for i in range(0, len(data), 16):
            yield data[i:i + 16]


def _get_segments(input_path, num_segments):
    """Split file into byte-range segments aligned to newline boundaries.
    Workers will seek directly to their segment — no data duplication."""
    file_size = os.path.getsize(input_path)
    if file_size == 0:
        return []
    seg_size = file_size // num_segments
    boundaries = [0]
    with open(input_path, "rb") as f:
        for i in range(1, num_segments):
            f.seek(min(i * seg_size, file_size))
            f.readline()  # advance to next line boundary
            pos = f.tell()
            if pos < file_size and pos != boundaries[-1]:
                boundaries.append(pos)
    boundaries.append(file_size)
    return [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]


def _worker(args):
    """Each worker opens the file independently and seeks to its segment.
    No data crosses IPC — only temp file paths are returned."""
    input_path, start, end, temp_dir, chunk_size = args
    temp_paths = []
    chunk = []
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
                chunk.append(ipv6_to_bytes(line))
                if len(chunk) >= chunk_size:
                    temp_paths.append(flush_chunk(chunk, temp_dir))
                    chunk.clear()
                    gc.collect()  # release memory arenas back to OS
    if chunk:
        temp_paths.append(flush_chunk(chunk, temp_dir))
    return temp_paths


def task1(input_path, output_path):
    temp_dir = tempfile.mkdtemp()

    # Limit workers: min of (available CPUs, what fits in memory budget)
    max_by_cpu = max(1, cpu_count() - 1)
    max_by_mem = max(1, MEMORY_BUDGET // (MIN_CHUNK * BYTES_PER_RECORD))
    num_workers = min(max_by_cpu, max_by_mem)

    # Distribute memory budget evenly; each worker sorts independently
    chunk_per_worker = MEMORY_BUDGET // (num_workers * BYTES_PER_RECORD)
    chunk_per_worker = max(MIN_CHUNK, chunk_per_worker)

    # Phase 1: parallel read + parse + sort into binary temp files
    segments = _get_segments(input_path, num_workers)

    if not segments:
        with open(output_path, "w") as out:
            out.write("0\n")
        os.rmdir(temp_dir)
        return

    worker_args = [
        (input_path, start, end, temp_dir, chunk_per_worker)
        for start, end in segments
    ]

    with Pool(num_workers) as pool:
        results = pool.map(_worker, worker_args)

    temp_files = []
    for paths in results:
        temp_files.extend(paths)

    if not temp_files:
        with open(output_path, "w") as out:
            out.write("0\n")
        os.rmdir(temp_dir)
        return

    # Phase 2: k-way merge via min-heap, count transitions = unique count
    buf_per_file = max(256, min(READ_BUF, 5_000_000 // max(len(temp_files), 1)))
    file_handles = [open(p, "rb") for p in temp_files]
    iterators = [sorted_records(fh, buf_per_file) for fh in file_handles]

    unique_count = 0
    prev = None
    # heapq.merge: O(log k) per element, implemented in C
    for val in heapq.merge(*iterators):
        if val != prev:
            unique_count += 1
            prev = val

    # Cleanup all temp artifacts
    for fh in file_handles:
        fh.close()
    for p in temp_files:
        os.remove(p)
    os.rmdir(temp_dir)

    with open(output_path, "w") as out:
        out.write(str(unique_count) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: {} <input_file> <output_file>".format(sys.argv[0]))
        sys.exit(1)
    task1(sys.argv[1], sys.argv[2])

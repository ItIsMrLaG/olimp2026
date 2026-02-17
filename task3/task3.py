import csv
import hashlib
import sys
import time
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count

RU_LOWER = "абвгдежзийклмнопрстуфхцчшщъыьэюя"
RU_UPPER = RU_LOWER.upper()
RU_SIZE = len(RU_LOWER)  # 32
EN_SIZE = 26

ADDRESS_MARKERS = ["ул.", "пл.", "пр.", "пер.", "наб.", "бул.", "ш."]
REQUIRED_MARKERS = [" д.", "кв."]

# Phone brute-force settings
PHONE_PREFIXES = ["8", "+7", "7"]
OPERATOR_RANGE = range(900, 1000)


# --- Caesar ciphers ---


def _caesar_ru(text: str, shift: int) -> str:
    """Shift Russian letters by `shift` positions; leave other chars as-is."""
    out = []
    for ch in text:
        idx = RU_LOWER.find(ch)
        if idx >= 0:
            out.append(RU_LOWER[(idx + shift) % RU_SIZE])
            continue

        idx = RU_UPPER.find(ch)
        if idx >= 0:
            out.append(RU_UPPER[(idx + shift) % RU_SIZE])
            continue

        out.append(ch)
    return "".join(out)


def _caesar_en(text: str, shift: int) -> str:
    """Shift Latin letters by `shift` positions; leave other chars as-is."""
    out = []
    for ch in text:
        o = ord(ch)
        if 97 <= o <= 122:
            out.append(chr((o - 97 + shift) % EN_SIZE + 97))
        elif 65 <= o <= 90:
            out.append(chr((o - 65 + shift) % EN_SIZE + 65))
        else:
            out.append(ch)
    return "".join(out)


# --- Shift detection ---


def _score_address(text: str) -> int:
    """Count how many known address markers appear in decrypted text."""
    lower = text.lower()
    score = 0

    for m in REQUIRED_MARKERS:
        if m in lower:
            score += 10

    for m in ADDRESS_MARKERS:
        if m in lower:
            score += 5

    return score


def _find_shift(encrypted_address: str) -> int:
    """Brute-force all 32 shifts, return the one producing the best address."""
    best_shift = 0
    best_score = -1

    for shift in range(RU_SIZE):
        decrypted = _caesar_ru(encrypted_address, -shift)
        score = _score_address(decrypted)
        if score > best_score:
            best_score = score
            best_shift = shift

    return best_shift


# --- SHA-1 phone brute-force ---


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


def _search_chunk(args: tuple) -> dict[str, str]:
    """Try all numbers for a given prefix and operator code range."""
    target_hashes, prefix, op_start, op_end = args
    target_set = set(target_hashes)
    found = {}

    for op in range(op_start, op_end):
        for num in range(10_000_000):
            phone = f"{prefix}{op}{num:07d}"
            h = _sha1(phone)
            if h in target_set:
                found[h] = phone
                print(f"  [hit] {h} -> {phone}")
                if len(found) == len(target_set):
                    return found

    return found


def bruteforce_phones(
    target_hashes: list[str],
    workers = None,
) -> dict[str, str]:
    """Brute-force SHA-1 hashes of Russian phone numbers."""
    if workers is None:
        workers = max(1, cpu_count() - 1)

    found: dict[str, str] = {}
    remaining = list(target_hashes)

    for prefix in PHONE_PREFIXES:
        if not remaining:
            break

        print(f"[*] Prefix '{prefix}', {len(remaining)} hashes left...")
        start = time.time()

        codes = list(OPERATOR_RANGE)
        chunk_size = max(1, len(codes) // workers)
        tasks = []
        for i in range(0, len(codes), chunk_size):
            batch = codes[i : i + chunk_size]
            tasks.append((remaining, prefix, batch[0], batch[-1] + 1))

        with Pool(workers) as pool:
            results = pool.map(_search_chunk, tasks)

        for result in results:
            found.update(result)

        remaining = [h for h in target_hashes if h not in found]
        elapsed = time.time() - start
        print(f"    {len(found)}/{len(target_hashes)} found ({elapsed:.1f}s)")

    return found


# --- Row processing ---


@dataclass
class DecodedRow:
    """Single row with all decryption results."""

    phone_hash: str
    phone_dec: str
    email_enc: str
    email_dec: str
    address_enc: str
    address_dec: str
    shift: int


# --- I/O ---


def process_csv(input_path: str, output_path: str) -> None:
    """Full pipeline: read CSV, crack phones, decrypt addresses and emails."""
    with open(input_path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    data_rows = [r for r in rows if len(r) >= 3 and any(c.strip() for c in r)]

    # Phase 1: Caesar (instant)
    print(f"[*] Decrypting {len(data_rows)} addresses and emails...")
    decoded: list[DecodedRow] = []
    for row in data_rows:
        phone, email, address = row[0].strip(), row[1].strip(), row[2].strip()
        shift = _find_shift(address)
        decoded.append(DecodedRow(
            phone_hash=phone,
            phone_dec="",
            email_enc=email,
            email_dec=_caesar_en(email, -shift),
            address_enc=address,
            address_dec=_caesar_ru(address, -shift),
            shift=shift,
        ))
    print(f"    done, shifts: {[d.shift for d in decoded]}")

    # Phase 2: SHA-1 brute-force (minutes)
    hashes = [d.phone_hash for d in decoded]
    print(f"[*] Brute-forcing {len(hashes)} phone hashes...")
    found = bruteforce_phones(hashes)
    for d in decoded:
        d.phone_dec = found.get(d.phone_hash, "")

    # Write output
    out_header = header + ["сдвиг", "Телефон_расшифровка", "Адрес_расшифровка", "email_расшифровка"]
    out_rows = [out_header]
    for d in decoded:
        out_rows.append([
            d.phone_hash,
            d.email_enc,
            d.address_enc,
            str(d.shift),
            d.phone_dec,
            d.address_dec,
            d.email_dec,
        ])

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(out_rows)

    recovered = sum(1 for d in decoded if d.phone_dec)
    print(f"\n[ok] {output_path} saved ({len(decoded)} rows, {recovered}/{len(decoded)} phones recovered)")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        process_csv(sys.argv[1], sys.argv[2])
    else:
        print(f"Usage: {sys.argv[0]} <input.csv> <output.csv>")
        sys.exit(1)


import math
import time
import concurrent.futures
import multiprocessing
from typing import List, Optional
from gmpy2 import mpz

# Constants
ARRAY_SIZE = 2000          # 8000:3500 = 1881s
FACTOR_UP_TO = 1000         # 4000:1500 = 201s; 4000:3000 = 199s; 4000:3500 = 201s
ORIGINS = [5, 7, 11, 13]   # 2000:500 = 24.8s; 2000:1000 = 24.7s; 2000:1500 = 25.2s; 2000:2000 = 25.7s; 2000:5000 = 32.7s
OFFSET = 120

# Base Cycles from your Java logic
BASE_CYCLES = {
    5: [[6, 8, 30, 32, 38, 48, 56, 62, 72, 78, 80, 86, 96, 102, 110, 120], [6, 14, 24, 30, 32, 54, 56, 62, 72, 80, 86, 96, 102, 104, 110, 120], [6, 8, 14, 24, 30, 38, 48, 54, 56, 78, 80, 86, 96, 104, 110, 120], [8, 14, 24, 30, 32, 38, 48, 54, 62, 72, 78, 80, 102, 104, 110, 120]],
    7: [[10, 16, 18, 40, 42, 48, 58, 66, 72, 82, 88, 90, 96, 106, 112, 120], [10, 16, 24, 34, 40, 42, 64, 66, 72, 82, 90, 96, 106, 112, 114, 120], [10, 16, 18, 24, 34, 40, 48, 58, 64, 66, 88, 90, 96, 106, 114, 120], [10, 18, 24, 34, 40, 42, 48, 58, 64, 72, 82, 88, 90, 112, 114, 120]],
    11: [[2, 8, 18, 26, 32, 42, 48, 50, 56, 66, 72, 80, 90, 96, 98, 120], [2, 24, 26, 32, 42, 50, 56, 66, 72, 74, 80, 90, 96, 104, 114, 120], [8, 18, 24, 26, 48, 50, 56, 66, 74, 80, 90, 96, 98, 104, 114, 120], [2, 8, 18, 24, 32, 42, 48, 50, 72, 74, 80, 90, 98, 104, 114, 120]],
    13: [[6, 16, 22, 30, 40, 46, 48, 70, 72, 78, 88, 96, 102, 112, 118, 120], [6, 16, 22, 24, 30, 40, 46, 54, 64, 70, 72, 94, 96, 102, 112, 120], [6, 16, 24, 30, 40, 46, 48, 54, 64, 70, 78, 88, 94, 96, 118, 120], [22, 24, 30, 40, 48, 54, 64, 70, 72, 78, 88, 94, 102, 112, 118, 120]]
}

def is_prime(n: int) -> bool:
    """Java's isPrime logic replacing the Miller-Rabin test."""
    if n < 2: return False
    if n == 2: return True
    if n % 2 == 0: return False
    sqrt_n = int(math.sqrt(n))
    for i in range(3, sqrt_n + 1, 2):
        if n % i == 0:
            return False
    return True

def lucas_lehmer(p: int) -> Optional[int]:
    """Returns p if M_p is prime using Lucas-Lehmer confirmation logic."""
    if p == 2: return p
    s = mpz(4)
    base = mpz(1) << p
    mask = base - 1
    for _ in range(p - 2):
        s = s * s - 2
        while s >= base:
            s = (s & mask) + (s >> p)
    return p if (s == mask or s == 0) else None

def get_base_cycle(p: int) -> List[int]:
    origin = (p - 5) % 12 + 5
    last_digit = p % 10
    mapping = {1: 0, 3: 1, 7: 2, 9: 3}
    idx = mapping.get(last_digit, 3)
    return BASE_CYCLES[origin][idx]

def has_no_small_factor(p: int) -> Optional[int]:
    """Returns p if NO factor found. Exits if factor >= 2^p - 1."""
    base_cycle = get_base_cycle(p)
    limit = (mpz(1) << p) - 1
    
    for i in range(FACTOR_UP_TO):
        p_offset_base = p * i * OFFSET
        for offset in base_cycle:
            factor = p_offset_base + p * offset + 1
            if factor >= limit:
                return p
            if pow(2, p, factor) == 1:
                return None
    return p

if __name__ == "__main__":
    start_total = time.perf_counter()
    num_cores = multiprocessing.cpu_count()
    
    # 1. Generate prime exponents using trial division
    candidates = []
    for start_val in ORIGINS:
        for i in range(ARRAY_SIZE):
            p = start_val + 12 * i
            if p > 3 and is_prime(p):
                candidates.append(p)
    candidates.sort()

    print(f"Testing on {num_cores} cores.")
    print(f"Initial Candidates: {len(candidates)}")
    
    # 2. Parallel Trial Division (Factoring)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(has_no_small_factor, candidates))
    
    survivors = [p for p in results if p is not None]
    print(f"Survivors after Trial Division: {len(survivors)}")
    print("-" * 65)

    # 3. Parallel Lucas-Lehmer Test
    primes_found = 0
    with concurrent.futures.ProcessPoolExecutor() as executor:
        ll_results = list(executor.map(lucas_lehmer, survivors))
    
    for p in ll_results:
        if p is not None:
            primes_found += 1
            print(f"{p:<6} | PRIME FOUND!")

    end_total = time.perf_counter()
    print("-" * 65)
    print(f"Total Primes: {primes_found}")
    print(f"Total Execution Time: {end_total - start_total:.4f}s")
    print("Mersenne Primes: 2^2-1 & 2^3-1 are skipped in the exponent generation algorithm")

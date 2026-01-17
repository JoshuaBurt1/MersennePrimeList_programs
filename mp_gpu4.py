import math
import time
import threading
import queue
import concurrent.futures
from typing import List, Optional
from gmpy2 import mpz, mul, sub, is_prime
import cupy as cp

# Next step is to move Lucas-Lehmer to GPU & try a Linux implementation
ARRAY_SIZE = 8000           # Total range per batch
CHUNK_SIZE = 1000           # Primes per GPU kernel launch
FACTOR_UP_TO = 200000       # 2000:15000 = 23.1s; 2000:20000 = 22.6s; 2000:30000 = 23.8s;
ORIGINS = [5, 7, 11, 13]    # 4000:100000 = 167.2s; 4000:200000 = 166.8s; # 4000:300000 = 199.8s
OFFSET = 120                # mp_gpu1 -> 8000:15000 = 1742s = 29m; mp_gpu4 -> 8000:200000 = 1562.6 = 26m
                            # mp_gpu1 -> 10000:20000 = 4360s = 72m    (5013 checked by LL)
                            # mp_gpu1 -> 12000:300000 = 7411s = 123m  (5414 checked by LL)
# Constant GPU data
CYCLE_DATA = cp.array([
    [[6, 8, 30, 32, 38, 48, 56, 62, 72, 78, 80, 86, 96, 102, 110, 120], [6, 14, 24, 30, 32, 54, 56, 62, 72, 80, 86, 96, 102, 104, 110, 120], [6, 8, 14, 24, 30, 38, 48, 54, 56, 78, 80, 86, 96, 104, 110, 120], [8, 14, 24, 30, 32, 38, 48, 54, 62, 72, 78, 80, 102, 104, 110, 120]],
    [[10, 16, 18, 40, 42, 48, 58, 66, 72, 82, 88, 90, 96, 106, 112, 120], [10, 16, 24, 34, 40, 42, 64, 66, 72, 82, 90, 96, 106, 112, 114, 120], [10, 16, 18, 24, 34, 40, 48, 58, 64, 66, 88, 90, 96, 106, 114, 120], [10, 18, 24, 34, 40, 42, 48, 58, 64, 72, 82, 88, 90, 112, 114, 120]],
    [[2, 8, 18, 26, 32, 42, 48, 50, 56, 66, 72, 80, 90, 96, 98, 120], [2, 24, 26, 32, 42, 50, 56, 66, 72, 74, 80, 90, 96, 104, 114, 120], [8, 18, 24, 26, 48, 50, 56, 66, 74, 80, 90, 96, 98, 104, 114, 120], [2, 8, 18, 24, 32, 42, 48, 50, 72, 74, 80, 90, 98, 104, 114, 120]],
    [[6, 16, 22, 30, 40, 46, 48, 70, 72, 78, 88, 96, 102, 112, 118, 120], [6, 16, 22, 24, 30, 40, 46, 54, 64, 70, 72, 94, 96, 102, 112, 120], [6, 16, 24, 30, 40, 46, 48, 54, 64, 70, 78, 88, 94, 96, 118, 120], [22, 24, 30, 40, 48, 54, 64, 70, 72, 78, 88, 94, 102, 112, 118, 120]]
], dtype=cp.uint64)

# Mega-Kernel (same high-performance version)
mega_kernel = cp.RawKernel(r'''
extern "C" __global__
void mega_sieve(const unsigned long long* candidates, const unsigned long long* cycles, 
                const unsigned int* o_idxs, const unsigned int* d_idxs, 
                bool* is_eliminated, int num_candidates, int factor_limit) {
    int tid_p = blockIdx.x * blockDim.x + threadIdx.x; 
    int tid_f = blockIdx.y * blockDim.y + threadIdx.y; 

    if (tid_p < num_candidates && tid_f < factor_limit) {
        unsigned long long p = candidates[tid_p];
        auto mul_mod = [](unsigned long long a, unsigned long long b, unsigned long long m) {
            unsigned long long hi = __umul64hi(a, b);
            unsigned long long lo = a * b;
            double q_est = ((double)hi * 18446744073709551616.0 + (double)lo) * (1.0 / (double)m);
            unsigned long long res = lo - (unsigned long long)q_est * m;
            while ((long long)res < 0) res += m;
            while (res >= m) res -= m;
            return res;
        };

        for (int j = 0; j < 16; j++) {
            unsigned long long f = (p * tid_f * 120) + (p * cycles[o_idxs[tid_p]*64 + d_idxs[tid_p]*16 + j]) + 1;
            unsigned long long res = 1, base = 2, exp = p;
            while (exp > 0) {
                if (exp & 1) res = mul_mod(res, base, f);
                base = mul_mod(base, base, f);
                exp >>= 1;
            }
            if (res == 1) { is_eliminated[tid_p] = true; break; }
        }
    }
}
''', 'mega_sieve')

def lucas_lehmer_optimized(p: int) -> Optional[int]:
    if p == 2: return p
    s = mpz(4); base = mpz(1) << p; mask = base - 1
    for _ in range(p - 2):
        s = mul(s, s); s = sub(s, 2)
        while s >= base: s = (s & mask) + (s >> p)
    return p if (s == mask or s == 0) else None

def gpu_worker(candidates: List[int], survivor_queue: queue.Queue):
    """Thread that feeds the GPU and pushes survivors to the CPU queue."""
    o_map, d_map = {5: 0, 7: 1, 11: 2, 13: 3}, {1: 0, 3: 1, 7: 2, 9: 3}
    
    for i in range(0, len(candidates), CHUNK_SIZE):
        batch = candidates[i:i+CHUNK_SIZE]
        num_p = len(batch)
        c_gpu = cp.array(batch, dtype=cp.uint64)
        o_idxs = cp.array([o_map[(p-5)%12+5] for p in batch], dtype=cp.uint32)
        d_idxs = cp.array([d_map.get(p%10, 3) for p in batch], dtype=cp.uint32)
        eliminated = cp.zeros(num_p, dtype=cp.bool_)

        mega_kernel((math.ceil(num_p/16), math.ceil(FACTOR_UP_TO/16)), (16, 16), 
                   (c_gpu, CYCLE_DATA, o_idxs, d_idxs, eliminated, num_p, FACTOR_UP_TO))
        
        survivors = [batch[j] for j in range(num_p) if not eliminated[j]]
        if survivors:
            survivor_queue.put(survivors)
            
    survivor_queue.put(None)

if __name__ == "__main__":
    start_total = time.perf_counter()
    candidates = sorted([p for start in ORIGINS for p in [start + 12*i for i in range(ARRAY_SIZE)] if p > 3 and is_prime(p)])
    print(f"Total Candidates: {len(candidates)}")

    survivor_queue = queue.Queue()
    gpu_thread = threading.Thread(target=gpu_worker, args=(candidates, survivor_queue))
    gpu_thread.start()

    # CPU Testing starts as soon as the first GPU batch is done
    with concurrent.futures.ProcessPoolExecutor() as executor:
        while True:
            batch = survivor_queue.get()
            if batch is None: break
            
            # Map LL test to available CPU cores
            results = list(executor.map(lucas_lehmer_optimized, batch))
            for r in results:
                if r: print(f"{r:<6} | PRIME FOUND!")
    
    gpu_thread.join()
    print(f"\nTotal Search Time: {time.perf_counter() - start_total:.4f}s")
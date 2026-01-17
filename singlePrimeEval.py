# Source: https://math.stackexchange.com/questions/3289191/can-my-home-pc-handle-the-lucas-lehmer-test-by-itself-or-do-i-need-gimps
# Maybe convert to C

import time
import sys
from functools import partial
from gmpy2 import is_strong_prp, mpz


def lucas_lehmer(p: mpz) -> bool:
    s = mpz(4)
    base = mpz(1 << p)
    mask = base - 1
    for i in range(p - 2):
        s = s*s - 2
        while s >= base:
            # s % 2**p - 1 is the digit sum base 2**p
            s = (s&mask) + (s >> p)
    return s==mask


# first check that p is prime using miller-rabin, then Lucas-Lehmer test
def main():
    p = mpz(sys.argv[1])
    bases = 2, 7, 61

    start_mr = time.time()
    if all(map(partial(is_strong_prp, p), bases)):
        print('PRIME')
    else:
        print('COMPOSITE')
    end_mr = time.time()
    mr_duration = end_mr - start_mr
    print(f"Miller-Rabin Time: {mr_duration:.6f} seconds")

    start_ll = time.time()
    if lucas_lehmer(p):
        print('PRIME')
    else:
        print('COMPOSITE')
    end_ll = time.time()
    ll_duration = end_ll - start_ll
    print(f"Lucas-Lehmer Time: {ll_duration:.6f} seconds")

if __name__ == '__main__':
    main()
#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#pragma warning(disable: 4146) 
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <gmp.h>
#include <omp.h>

#define ARRAY_SIZE 1000
#define FACTOR_UP_TO 375
#define OFFSET 120

int BASE_CYCLES[14][4][16] = { 0 };

void init_base_cycles() {
    int c5[4][16] = { {6,8,30,32,38,48,56,62,72,78,80,86,96,102,110,120},{6,14,24,30,32,54,56,62,72,80,86,96,102,104,110,120},{6,8,14,24,30,38,48,54,56,78,80,86,96,104,110,120},{8,14,24,30,32,38,48,54,62,72,78,80,102,104,110,120} };
    int c7[4][16] = { {10,16,18,40,42,48,58,66,72,82,88,90,96,106,112,120},{10,16,24,34,40,42,64,66,72,82,90,96,106,112,114,120},{10,16,18,24,34,40,48,58,64,66,88,90,96,106,114,120},{10,18,24,34,40,42,48,58,64,72,82,88,90,112,114,120} };
    int c11[4][16] = { {2,8,18,26,32,42,48,50,56,66,72,80,90,96,98,120},{2,24,26,32,42,50,56,66,72,74,80,90,96,104,114,120},{8,18,24,26,48,50,56,66,74,80,90,96,98,104,114,120},{2,8,18,24,32,42,48,50,72,74,80,90,98,104,114,120} };
    int c13[4][16] = { {6,16,22,30,40,46,48,70,72,78,88,96,102,112,118,120},{6,16,22,24,30,40,46,54,64,70,72,94,96,102,112,120},{6,16,24,30,40,46,48,54,64,70,78,88,94,96,118,120},{22,24,30,40,48,54,64,70,72,78,88,94,102,112,118,120} };
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 16; j++) {
            BASE_CYCLES[5][i][j] = c5[i][j]; BASE_CYCLES[7][i][j] = c7[i][j];
            BASE_CYCLES[11][i][j] = c11[i][j]; BASE_CYCLES[13][i][j] = c13[i][j];
        }
    }
}

bool is_prime(int n) {
    if (n < 2) return false;
    if (n == 2) return true;
    if (n % 2 == 0) return false;
    int limit = (int)sqrt((double)n);
    for (int i = 3; i <= limit; i += 2) if (n % i == 0) return false;
    return true;
}

int lucas_lehmer_optimized(int p, mpz_t s, mpz_t base, mpz_t mask, mpz_t temp) {
    if (p == 2) return p;

    mpz_set_ui(s, 4);
    mpz_set_ui(base, 1);
    mpz_mul_2exp(base, base, p);
    mpz_sub_ui(mask, base, 1);

    for (int i = 0; i < p - 2; i++) {
        mpz_mul(s, s, s);
        mpz_sub_ui(s, s, 2);

        // Fast reduction for Mersenne numbers
        while (mpz_cmp(s, mask) > 0) {
            mpz_fdiv_q_2exp(temp, s, p);
            mpz_and(s, s, mask);
            mpz_add(s, s, temp);
        }
        if (mpz_cmp(s, mask) == 0) mpz_set_ui(s, 0);
    }
    return (mpz_sgn(s) == 0) ? p : 0;
}

int main() {
    init_base_cycles();

    // Set threads to 22 explicitly to match your Python performance
    omp_set_num_threads(22);

    double start_time = omp_get_wtime();

    int origins[] = { 5, 7, 11, 13 };
    int* candidates = (int*)malloc(ARRAY_SIZE * 4 * sizeof(int));
    int cand_count = 0;

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < ARRAY_SIZE; j++) {
            int p = origins[i] + 12 * j;
            if (p > 3 && is_prime(p)) candidates[cand_count++] = p;
        }
    }

    printf("Testing on %d cores.\n", omp_get_max_threads());
    printf("Initial Candidates: %d\n", cand_count);

    // 1. Parallel Trial Division (Optimized)
#pragma omp parallel
    {
        mpz_t factor_mpz, two, p_mpz, res;
        mpz_inits(factor_mpz, two, p_mpz, res, NULL);
        mpz_set_ui(two, 2);

#pragma omp for schedule(dynamic, 1)
        for (int i = 0; i < cand_count; i++) {
            int p = candidates[i];
            int origin = (p - 5) % 12 + 5;
            int last_digit = p % 10;
            int idx = (last_digit == 1) ? 0 : (last_digit == 3) ? 1 : (last_digit == 7) ? 2 : 3;

            mpz_set_ui(p_mpz, (unsigned long)p);
            bool found_factor = false;

            for (int k = 0; k < FACTOR_UP_TO && !found_factor; k++) {
                uint64_t base_val = (uint64_t)p * k * OFFSET;
                for (int j = 0; j < 16; j++) {
                    // Use native uint64_t for arithmetic
                    uint64_t f_val = (uint64_t)p * BASE_CYCLES[origin][idx][j] + base_val + 1;

                    // Only convert to GMP for modular exponentiation
                    mpz_set_ui(factor_mpz, (unsigned long)f_val);

                    mpz_powm(res, two, p_mpz, factor_mpz);
                    if (mpz_cmp_ui(res, 1) == 0) {
                        found_factor = true;
                        break;
                    }
                }
            }
            if (found_factor) candidates[i] = 0;
        }
        mpz_clears(factor_mpz, two, p_mpz, res, NULL);
    }

    int* survivors = (int*)malloc(cand_count * sizeof(int));
    int survivor_count = 0;
    for (int i = 0; i < cand_count; i++) {
        if (candidates[i] != 0) survivors[survivor_count++] = candidates[i];
    }
    printf("Survivors: %d\n", survivor_count);
    printf("-----------------------------------------------------------------\n");

    // 2. Parallel Lucas-Lehmer
    int primes_found = 0;
#pragma omp parallel
    {
        mpz_t s_p, base_p, mask_p, temp_p;
        mpz_inits(s_p, base_p, mask_p, temp_p, NULL);

#pragma omp for schedule(dynamic, 1) reduction(+:primes_found)
        for (int i = 0; i < survivor_count; i++) {
            int result = lucas_lehmer_optimized(survivors[i], s_p, base_p, mask_p, temp_p);
            if (result != 0) {
#pragma omp critical
                printf("%-6d | PRIME FOUND!\n", result);
                primes_found++;
            }
        }
        mpz_clears(s_p, base_p, mask_p, temp_p, NULL);
    }

    printf("-----------------------------------------------------------------\n");
    printf("Total Primes: %d\n", primes_found);
    printf("Total Execution Time: %.4fs\n", omp_get_wtime() - start_time);

    free(candidates); free(survivors);
    return 0;
}
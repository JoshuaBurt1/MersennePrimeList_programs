#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>
#include <gmp.h>
#include <omp.h>

// Flattened CYCLE_DATA (4 * 4 * 16 = 256 elements)
__constant__ unsigned long long d_CYCLE_DATA[256] = {
    6, 8, 30, 32, 38, 48, 56, 62, 72, 78, 80, 86, 96, 102, 110, 120, 6, 14, 24, 30, 32, 54, 56, 62, 72, 80, 86, 96, 102, 104, 110, 120, 6, 8, 14, 24, 30, 38, 48, 54, 56, 78, 80, 86, 96, 104, 110, 120, 8, 14, 24, 30, 32, 38, 48, 54, 62, 72, 78, 80, 102, 104, 110, 120,
    10, 16, 18, 40, 42, 48, 58, 66, 72, 82, 88, 90, 96, 106, 112, 120, 10, 16, 24, 34, 40, 42, 64, 66, 72, 82, 90, 96, 106, 112, 114, 120, 10, 16, 18, 24, 34, 40, 48, 58, 64, 66, 88, 90, 96, 106, 114, 120, 10, 18, 24, 34, 40, 42, 48, 58, 64, 72, 82, 88, 90, 112, 114, 120,
    2, 8, 18, 26, 32, 42, 48, 50, 56, 66, 72, 80, 90, 96, 98, 120, 2, 24, 26, 32, 42, 50, 56, 66, 72, 74, 80, 90, 96, 104, 114, 120, 8, 18, 24, 26, 48, 50, 56, 66, 74, 80, 90, 96, 98, 104, 114, 120, 2, 8, 18, 24, 32, 42, 48, 50, 72, 74, 80, 90, 98, 104, 114, 120,
    6, 16, 22, 30, 40, 46, 48, 70, 72, 78, 88, 96, 102, 112, 118, 120, 6, 16, 22, 24, 30, 40, 46, 54, 64, 70, 72, 94, 96, 102, 112, 120, 6, 16, 24, 30, 40, 46, 48, 54, 64, 70, 78, 88, 94, 96, 118, 120, 22, 24, 30, 40, 48, 54, 64, 70, 72, 78, 88, 94, 102, 112, 118, 120
};

__global__ void mega_sieve(const unsigned long long* candidates, 
                           const unsigned int* o_idxs, 
                           const unsigned int* d_idxs, 
                           bool* is_eliminated, 
                           int num_p, int factor_limit) {
    int tid_p = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_f = blockIdx.y * blockDim.y + threadIdx.y;

    if (tid_p < num_p && tid_f < factor_limit) {
        if (is_eliminated[tid_p]) return;
        unsigned long long p = candidates[tid_p];
        unsigned int base_idx = (o_idxs[tid_p] * 64) + (d_idxs[tid_p] * 16);

        for (int j = 0; j < 16; j++) {
            unsigned long long f = (p * (unsigned long long)tid_f * 120) + (p * d_CYCLE_DATA[base_idx + j]) + 1;
            unsigned long long res = 1, base = 2, exp = p;
            
            while (exp > 0) {
                if (exp & 1) res = (unsigned long long)((unsigned __int128)res * base % f);
                base = (unsigned long long)((unsigned __int128)base * base % f);
                exp >>= 1;
            }
            if (res == 1) { is_eliminated[tid_p] = true; break; }
        }
    }
}

bool lucas_lehmer(unsigned long p) {
    if (p == 2) return true;
    mpz_t s, base, mask, temp;
    mpz_inits(s, base, mask, temp, NULL);
    mpz_set_ui(s, 4);
    mpz_set_ui(base, 1);
    mpz_mul_2exp(base, base, p); 
    mpz_sub_ui(mask, base, 1);

    for (unsigned long i = 0; i < p - 2; i++) {
        mpz_mul(s, s, s);
        mpz_sub_ui(s, s, 2);
        while (mpz_cmp(s, base) >= 0) {
            mpz_tdiv_q_2exp(temp, s, p);
            mpz_and(s, s, mask);
            mpz_add(s, s, temp);
        }
    }
    bool result = (mpz_cmp_ui(s, 0) == 0 || mpz_cmp(s, mask) == 0);
    mpz_clears(s, base, mask, temp, NULL);
    return result;
}

int main() {
    const int ARRAY_SIZE = 2000;
    const int FACTOR_UP_TO = 30000;
    std::vector<int> origins = {5, 7, 11, 13};
    std::vector<unsigned long long> h_candidates;

    for (int start : origins) {
        for (int i = 0; i < ARRAY_SIZE; i++) {
            unsigned long long p = start + 12 * i;
            if (p > 3) {
                mpz_t g_p; mpz_init_set_ui(g_p, p);
                if (mpz_probab_prime_p(g_p, 25)) h_candidates.push_back(p);
                mpz_clear(g_p);
            }
        }
    }
    std::sort(h_candidates.begin(), h_candidates.end());
    int num_p = h_candidates.size();
    std::cout << "Total Candidates: " << num_p << std::endl;

    std::vector<unsigned int> h_o_idxs(num_p), h_d_idxs(num_p);
    for (int i = 0; i < num_p; i++) {
        h_o_idxs[i] = (h_candidates[i] - 5) % 12 == 0 ? 0 : (h_candidates[i] - 7) % 12 == 0 ? 1 : (h_candidates[i] - 11) % 12 == 0 ? 2 : 3;
        unsigned int last_digit = h_candidates[i] % 10;
        h_d_idxs[i] = (last_digit == 1) ? 0 : (last_digit == 3) ? 1 : (last_digit == 7) ? 2 : 3;
    }

    unsigned long long *d_cand; unsigned int *d_o, *d_d; bool *d_elim;
    cudaMalloc(&d_cand, num_p * sizeof(unsigned long long));
    cudaMalloc(&d_o, num_p * sizeof(unsigned int));
    cudaMalloc(&d_d, num_p * sizeof(unsigned int));
    cudaMalloc(&d_elim, num_p * sizeof(bool));
    cudaMemset(d_elim, 0, num_p * sizeof(bool));

    cudaMemcpy(d_cand, h_candidates.data(), num_p * sizeof(unsigned long long), cudaMemcpyHostToDevice);
    cudaMemcpy(d_o, h_o_idxs.data(), num_p * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_d, h_d_idxs.data(), num_p * sizeof(unsigned int), cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((num_p + 15) / 16, (FACTOR_UP_TO + 15) / 16);
    
    // --- TIME GPU ---
    auto start_gpu = std::chrono::high_resolution_clock::now();
    mega_sieve<<<blocks, threads>>>(d_cand, d_o, d_d, d_elim, num_p, FACTOR_UP_TO);
    cudaDeviceSynchronize();
    auto end_gpu = std::chrono::high_resolution_clock::now();

    std::vector<char> h_elim(num_p);
    cudaMemcpy(h_elim.data(), d_elim, num_p * sizeof(bool), cudaMemcpyDeviceToHost);

    // --- TIME CPU ---
    int survivors = 0;
    auto start_cpu = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for reduction(+:survivors) schedule(dynamic)
    for (int i = 0; i < num_p; i++) {
        if (!h_elim[i]) {
            survivors++;
            if (lucas_lehmer(h_candidates[i])) {
                #pragma omp critical
                std::cout << "PRIME (p-value): " << h_candidates[i] << std::endl;
            }
        }
    }
    auto end_cpu = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> gpu_dur = end_gpu - start_gpu;
    std::chrono::duration<double> cpu_dur = end_cpu - start_cpu;

    std::cout << "\n--- Statistics ---" << std::endl;
    std::cout << "GPU Sieve Time: " << gpu_dur.count() << "s" << std::endl;
    std::cout << "CPU LL Time:    " << cpu_dur.count() << "s" << std::endl;
    std::cout << "Survivors:      " << survivors << std::endl;

    cudaFree(d_cand); cudaFree(d_o); cudaFree(d_d); cudaFree(d_elim);
    return 0;
}


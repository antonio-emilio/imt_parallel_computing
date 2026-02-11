#include <vector>
#include <iostream>
#include <cstring>
#include <chrono>
#include <cmath>
#include <atomic>
#include <numeric>
#include <new>

#include <string>

#include <omp.h>

#define USE_128_BITS 1

#define SERIAL 0
#define PARFOR_NAIVE 1
#define PARFOR_PARTIAL_SUMS 2
#define PARFOR_ATOMIC 3
#define PARFOR_REDUCE 4

#define STRATEGY SERIAL

// PaddedElement structure to align data in memory and avoid
// cache line false sharing
template <typename T>
struct alignas(std::hardware_destructive_interference_size) PaddedElement {
    T value;
};

// Function to convert an unsigned __int128 to a string
std::string uint128_to_str(unsigned __int128 n)
{
    if(n == 0) return "0";
    std::string s;
    while (n != 0)
    {
        s.insert(0, 1, char(n%10+'0'));
        n /= 10;
    }
    return s;
}

// Overload to print unsigned __int128 number the C++ way
std::ostream& operator<<(std::ostream& o, unsigned __int128 n)
{
    return o << uint128_to_str(n);
}

// Function used to switch between integer powers
template<size_t exponent, class uint_sum_t>
uint_sum_t power(uint_sum_t i)
{
    if constexpr (exponent == 1)
        return i;
    else if constexpr (exponent == 2)
        return i * i;
    else if constexpr (exponent == 3)
        return i * i * i;
    else
        static_assert(false, "Keep the exponent between 1 and 3!");
}

// Reference value for \sum_{i=1}^{N} i^k, k \in {1, 2, 3}
template<size_t exponent, class uint_sum_t>
uint_sum_t analytical_sum(uint_sum_t N)
{
    if constexpr (exponent == 1)
        return N * (N + 1) / 2;
    else if constexpr (exponent == 2)
        return N * (N + 1) * (2 * N + 1) / 6;
    else if constexpr (exponent == 3)
        return N * N * (N + 1) * (N + 1) / 4;
    else
        static_assert(false, "Keep the exponent between 1 and 3!");
}

template<size_t exponent, class uint_sum_t>
uint_sum_t compute_sum_of_powers(uint_sum_t N)
{
    uint_sum_t sum = 0;
#if STRATEGY == SERIAL
    // Example code with no parallelism at all
    for(size_t i = 1; i < N + 1; ++i)
        sum += power<exponent>(uint_sum_t{i});
#elif STRATEGY == PARFOR_NAIVE
    // TODO Use an omp for to parallelize the code
#elif STRATEGY == PARFOR_PARTIAL_SUMS
    // TODO Fix the naive parfor implementation using partial sums
    std::vector<PaddedElement<uint_sum_t>> partial_sums(omp_get_max_threads());

    // Accumulate the partial sums into the final sum
    sum = std::accumulate(partial_sums.cbegin(), partial_sums.cend(), uint_sum_t{0}, [](uint_sum_t acc, const PaddedElement<uint_sum_t>& el) { return acc + el.value; });
#elif STRATEGY == PARFOR_ATOMIC
    // TODO Fix the naive parfor implementation using an std::atomic
    static_assert(USE_128_BITS == 0, "Atomic is not supported for 128 bits integers. Switch USE_128_BITS to 0 for your atomic implementation only.");
    std::atomic<uint_sum_t> safe_sum{uint_sum_t{0}};
#elif STRATEGY == PARFOR_REDUCE
    // TODO Fix the naive parfor implementation using an omp reduction
#else
    std::cerr << "Not implemented." << std::endl;
#endif
    return sum;
}


int main(int argc, char* argv[])
{
    /*
     * Exponent you want to use for the summation
     */
    const constexpr size_t exponent = 3;

    /*
     * Check arguments and set things up
     */
    if(argc != 2) {
        std::cerr << "Usage: " << argv[0] << " N" << std::endl;
        return EXIT_FAILURE;
    }

    /*
     * Switch between 64 and 128 bits long unsigned integers
     */
#if USE_128_BITS
    using uint_sum_t = unsigned __int128;
#else
    using uint_sum_t = unsigned long long;
#endif

    uint_sum_t N = std::stoull(argv[1]);

    /*
     * Compute the sum and measure execution time
     */
    auto start = std::chrono::steady_clock::now();

    uint_sum_t sum = compute_sum_of_powers<3>(N);

    auto end = std::chrono::steady_clock::now();

    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";

    std::cout << "Computed result : " << sum << std::endl;
    std::cout << "Expected result : " << analytical_sum<exponent>(N) << std::endl;

    return EXIT_SUCCESS;
}


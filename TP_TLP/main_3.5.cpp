#include <omp.h>
#include <iostream>
#include <cstdlib>
#include <thread>

int main(int argc, char* argv[])
{
    if(argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " num_iter" << std::endl;
        return EXIT_FAILURE;
    }

    size_t num_iters = strtoull(argv[1], nullptr, 10);

    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for(size_t i = 0; i < num_iters; ++i)
        {
            #pragma omp critical
            {
                std::cout << "Thread " << omp_get_thread_num() << " going to sleep for " << i << "s\n";
            }
            std::this_thread::sleep_for(std::chrono::seconds{i});
        }

        #pragma omp critical
        {
            std::cout << "Thread " << omp_get_thread_num() << " done!\n";
        }
    }

    return EXIT_SUCCESS;
}


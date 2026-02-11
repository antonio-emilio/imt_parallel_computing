//OpenMP header
#include <omp.h>
#include <iostream>

int main()
{
    #pragma omp parallel
    {
        std::cerr<< "Hello World... from thread " << omp_get_thread_num() <<
                    " / " << omp_get_num_threads() << std::endl;
    }
    return EXIT_SUCCESS;
}


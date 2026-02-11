#include <omp.h>
#include <vector>
#include <iostream>
#include <cstring>
#include <chrono>
#include <cmath>

// Switching between different //ization strategies.

#define SERIAL 0
#define PARFOR 1
#define TASKS  2
#define TASKS_NAIVE  3

#define STRATEGY SERIAL

void compute_signal(std::vector<double>& signal, double A, double fm, double fp, double tmin, double dt)
{
#if STRATEGY == SERIAL
    for(size_t i = 0; i < signal.size(); ++i)
    {
        double t = tmin + i * dt;
        signal[i] = (1.0 + A * std::cos(2. * M_PI * fm * t)) * std::cos(2. * M_PI * fp * t);
    }
#elif STRATEGY == TASKS
// TODO Use tasks (fixed or dynamic number) to compute the signal
#elif STRATEGY == PARFOR
// TODO Use a parfor to compute the signal
#else
    std::cerr << "Not implemented!" << std::endl;
#endif
}

void plot_results(const std::vector<double>& samples, double tmin, double dt)
{
    FILE *gnuplot_pipe = popen("gnuplot -persistent", "w");
    fprintf(gnuplot_pipe, "set title 'Nice signal'\n");
    // fprintf(gnuplot_pipe, "set style data lp\n"); // Use to see markers
    fprintf(gnuplot_pipe, "set style data l\n");
    fprintf(gnuplot_pipe, "set xlabel 't'\n");
    fprintf(gnuplot_pipe, "set ylabel 's(t)'\n");
    // fprintf(gnuplot_pipe, "set term dumb\n"); // For fun
    fprintf(gnuplot_pipe, "plot '-'\n");
    for(size_t i = 0; i < samples.size(); ++i)
        fprintf(gnuplot_pipe, "%lf %lf\n", tmin + dt * i, samples[i]);
    fprintf(gnuplot_pipe, "e\n");
    pclose(gnuplot_pipe);
}

int main(int argc, char* argv[])
{
    /*
     * Check arguments and set things up
     */
    if(argc != 6) {
        std::cerr << "Usage: " << argv[0] << " N fm fp tmin tmax" << std::endl;
        return EXIT_FAILURE;
    }

    size_t num_samples = std::stoull(argv[1]);
    double fm = std::stod(argv[2]), fp = std::stod(argv[3]), tmin = std::stod(argv[4]), tmax = std::stod(argv[5]), A = 0.5;
    double dt = std::abs(tmax - tmin) / num_samples;

    /*
     * Compute the signal and measure execution time
     */
    std::vector<double> signal(num_samples, 100. * A);
    auto start = std::chrono::steady_clock::now();
    compute_signal(signal, A, fm, fp, tmin, dt);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout << "Computed signal in: " << elapsed_seconds.count() << "s" << std::endl;

    /*
     * Plot the signal
     */
    plot_results(signal, tmin, dt);
    
    return EXIT_SUCCESS;
}


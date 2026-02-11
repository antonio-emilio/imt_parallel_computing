#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <memory>
#include <cassert>
#include <complex>
#include <cmath>
#include <algorithm>

#include <omp.h>

using namespace std::complex_literals;

std::vector<double> compute_signal(size_t num_samples, double A, double fm, double fp, double dt, double tmin)
{
    std::vector<double> signal(num_samples, 10. * A);

    // TODO Fill this function with you favorite parallelization strategy

    return signal;
}

// TODO compute the DFT matrix. Matrices can be stored in flattened array
// or in arrays of arrays (std::vector<std::vector<double>>). You can change the
// prototype of this function according to what you prefer.
std::vector<std::complex<double>> compute_dft_matrix(size_t num_samples)
{
    // Compute the DFT matrix rows
    double N = num_samples;
    std::complex<double> w = std::exp(-2.0i * M_PI / N);

    // Print ahead of time how much memory we will need to store the
    // DFT matrix.
    std::cout << "Allocating the DFT matrix will require ~" << N * N * sizeof(decltype(w)) / 1e9
              << " GB of RAM." << std::endl; 

    std::vector<std::complex<double>> dft_matrix(num_samples * num_samples, 0.0);

    // TODO fill the dft matrix with a parallelized loop

    return dft_matrix;
}

std::vector<std::complex<double>> compute_dft_of_signal(const std::vector<std::complex<double>>& dft_matrix,
                                                        const std::vector<double>& signal)
{
    size_t N = signal.size();
    std::vector<std::complex<double>> dft_signal(N, 0.0);

    // TODO fill the dft vector in parallel using what you have learned in the previous exercises

    return dft_signal;
}

std::vector<double> pre_process_signal(const std::vector<double>& signal)
{
    // We multiply the time domain signal vector by (-1)^i to avoid doing
    // the fftshift after computing the DFT
    std::vector<double> pre_processed_signal(signal.size());
    #pragma omp parallel for
    for(size_t i = 0; i < signal.size(); ++i)
        pre_processed_signal[i] = signal[i] * std::pow(-1., i);
    return pre_processed_signal;
}

void plot_results(const std::vector<double>& samples, const std::vector<double>& dft,
                  double tmin, double dt)
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

    FILE *gnuplot_pipe_fft = popen("gnuplot -persistent", "w");
    fprintf(gnuplot_pipe_fft, "set title 'Spectrum of the nice signal'\n");
    // fprintf(gnuplot_pipe_fft, "set style data lp\n"); // Use to see markers
    fprintf(gnuplot_pipe_fft, "set style data l\n");
    fprintf(gnuplot_pipe_fft, "set xlabel 'f'\n");
    fprintf(gnuplot_pipe_fft, "set ylabel 'A(f)'\n");
    // fprintf(gnuplot_pipe_fft, "set term dumb\n"); // For fun
    fprintf(gnuplot_pipe_fft, "plot '-'\n");
    double df = 1.0 / dt / samples.size(), fmax = 1.0 / dt;
    for(size_t i = 0; i < dft.size(); ++i)
        fprintf(gnuplot_pipe_fft, "%lf %lf\n", df * i - fmax / 2. , dft[i]);
    fprintf(gnuplot_pipe_fft, "e\n");
    pclose(gnuplot_pipe_fft);
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

    std::vector<double> signal;
    std::vector<std::complex<double>> dft_matrix, dft_signal;

    { // Compute the signal
    auto start = std::chrono::steady_clock::now();

    signal = compute_signal(num_samples, A,  fm,  fp,  dt,  tmin);

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout << "Computed signal in: " << elapsed_seconds.count() << "s" << std::endl;
    }

    { // Compute the DFT matrix
    auto start = std::chrono::steady_clock::now();

    dft_matrix = compute_dft_matrix(num_samples);

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout << "Computed the dft matrix in: " << elapsed_seconds.count() << "s" << std::endl;
    }

    { // Compute the DFT of the signal
    auto start = std::chrono::steady_clock::now();

    dft_signal = compute_dft_of_signal(dft_matrix, pre_process_signal(signal));

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout << "Computed the dft of the signal in: " << elapsed_seconds.count() << "s" << std::endl;
    }

    // We only want the absolute value of the DFT coefs for convenience
    std::vector<double> abs_dft_signal(signal.size(), 0.0);
    std::transform(dft_signal.cbegin(), dft_signal.cend(), abs_dft_signal.begin(),
                   [](std::complex<double> v) {return std::abs(v);});

    // Plot the results!
    plot_results(signal, abs_dft_signal, tmin, dt);

    return EXIT_SUCCESS;
}


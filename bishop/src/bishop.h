#ifndef BISHOP_H
#define BISHOP_H

#include <vector>
#include <complex>
#include <Eigen/Dense>

class FourierLayer1D {
public:
    FourierLayer1D(int n_modes, int n_points);
    std::vector<double> forward(const std::vector<double>& input) const;
    int param_count() const { return 2 * n_modes_; }
    // access to parameters
    double get_weight_real(int i) const { return weight_[i].real(); }
    double get_weight_imag(int i) const { return weight_[i].imag(); }
    void set_weight_real(int i, double v) { weight_[i].real(v); }
    void set_weight_imag(int i, double v) { weight_[i].imag(v); }
    const std::complex<double>& weight(int i) const { return weight_[i]; }

private:
    int n_modes_;
    int n_points_;
    std::vector<std::complex<double>> weight_;
};

class NeuralOperator {
public:
    NeuralOperator(int n_modes, int n_points);
    std::vector<double> forward(const std::vector<double>& input) const;
    void train(const std::vector<std::vector<double>>& inputs,
               const std::vector<std::vector<double>>& targets,
               int epochs, double lr);
private:
    FourierLayer1D layer_;
    int n_points_;
};

#endif // BISHOP_H

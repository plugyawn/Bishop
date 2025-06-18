#include "bishop.h"
#include <cmath>
#include <iostream>

// Discrete Fourier Transform
static std::vector<std::complex<double>> dft(const std::vector<double>& input) {
    int N = input.size();
    std::vector<std::complex<double>> out(N);
    const double PI = std::acos(-1);
    for(int k=0;k<N;++k){
        std::complex<double> sum(0.0,0.0);
        for(int n=0;n<N;++n){
            double angle = -2*PI*k*n/N;
            sum += std::polar(input[n], angle);
        }
        out[k] = sum;
    }
    return out;
}

static std::vector<double> idft(const std::vector<std::complex<double>>& input) {
    int N = input.size();
    std::vector<double> out(N);
    const double PI = std::acos(-1);
    for(int n=0;n<N;++n){
        std::complex<double> sum(0.0,0.0);
        for(int k=0;k<N;++k){
            double angle = 2*PI*k*n/N;
            sum += input[k] * std::exp(std::complex<double>(0,angle));
        }
        out[n] = sum.real()/N;
    }
    return out;
}

FourierLayer1D::FourierLayer1D(int n_modes, int n_points)
    : n_modes_(n_modes), n_points_(n_points), weight_(n_modes, {1.0,0.0}) {}

std::vector<double> FourierLayer1D::forward(const std::vector<double>& input) const {
    auto freq = dft(input);
    for(int k=0;k<n_modes_;++k){
        freq[k] *= weight_[k];
        if(k>0) freq[freq.size()-k] *= std::conj(weight_[k]);
    }
    return idft(freq);
}

NeuralOperator::NeuralOperator(int n_modes, int n_points)
    : layer_(n_modes, n_points), n_points_(n_points) {}

std::vector<double> NeuralOperator::forward(const std::vector<double>& input) const {
    return layer_.forward(input);
}

void NeuralOperator::train(const std::vector<std::vector<double>>& inputs,
                           const std::vector<std::vector<double>>& targets,
                           int epochs, double lr) {
    const double eps = 1e-6;
    int n_params = layer_.param_count();
    for(int epoch=0; epoch<epochs; ++epoch){
        // compute current loss
        double loss = 0.0;
        for(size_t i=0;i<inputs.size();++i){
            auto pred = forward(inputs[i]);
            for(int j=0;j<n_points_;++j){
                double diff = pred[j]-targets[i][j];
                loss += diff*diff;
            }
        }
        loss /= inputs.size()*n_points_;
        std::cout << "epoch " << epoch << " loss=" << loss << std::endl;

        // compute gradient via finite difference
        std::vector<double> grad(n_params,0.0);
        for(int p=0;p<layer_.param_count()/2;++p){
            for(int part=0; part<2; ++part){
                double original = part==0 ? layer_.get_weight_real(p) : layer_.get_weight_imag(p);
                if(part==0) layer_.set_weight_real(p, original + eps);
                else layer_.set_weight_imag(p, original + eps);
                double loss_plus=0.0;
                for(size_t i=0;i<inputs.size();++i){
                    auto pred = forward(inputs[i]);
                    for(int j=0;j<n_points_;++j){
                        double diff = pred[j]-targets[i][j];
                        loss_plus += diff*diff;
                    }
                }
                loss_plus /= inputs.size()*n_points_;
                if(part==0) layer_.set_weight_real(p, original - eps);
                else layer_.set_weight_imag(p, original - eps);
                double loss_minus=0.0;
                for(size_t i=0;i<inputs.size();++i){
                    auto pred = forward(inputs[i]);
                    for(int j=0;j<n_points_;++j){
                        double diff = pred[j]-targets[i][j];
                        loss_minus += diff*diff;
                    }
                }
                loss_minus /= inputs.size()*n_points_;
                double g = (loss_plus - loss_minus)/(2*eps);
                grad[2*p + part] = g;
                if(part==0) layer_.set_weight_real(p, original);
                else layer_.set_weight_imag(p, original);
            }
        }
        // update
        for(int p=0;p<layer_.param_count()/2;++p){
            layer_.set_weight_real(p, layer_.get_weight_real(p) - lr*grad[2*p]);
            layer_.set_weight_imag(p, layer_.get_weight_imag(p) - lr*grad[2*p+1]);
        }
    }
}


#include "bishop.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>
#include <cstdlib>

static std::vector<double> generate_function(double freq, int n_points){
    std::vector<double> v(n_points);
    const double PI = std::acos(-1);
    for(int i=0;i<n_points;++i){
        double x = static_cast<double>(i)/n_points;
        v[i] = std::sin(2*PI*freq*x);
    }
    return v;
}

int main(){
    int N = 64; // discretization points
    int modes = 5;
    NeuralOperator net(modes, N);
    std::mt19937 gen(0);
    std::uniform_real_distribution<double> dist(0.5,2.0);
    std::vector<std::vector<double>> inputs, targets;
    for(int i=0;i<10;++i){
        double f = dist(gen);
        inputs.push_back(generate_function(f,N));
        targets.push_back(generate_function(2*f,N));
    }
    net.train(inputs, targets, 20, 0.01);
    auto input = generate_function(1.0,N);
    auto target = generate_function(2.0,N);
    auto pred = net.forward(input);
    std::ofstream out("plot.dat");
    for(int i=0;i<N;++i){
        double x = static_cast<double>(i)/N;
        out << x << " " << pred[i] << " " << target[i] << "\n";
    }
    out.close();
    std::ofstream gp("plot.gp");
    gp << "set terminal png size 800,600\n";
    gp << "set output 'plot.png'\n";
    gp << "plot 'plot.dat' using 1:2 with lines title 'prediction', 'plot.dat' using 1:3 with lines title 'target'\n";
    gp.close();
    std::system("gnuplot plot.gp");
    std::cout << "Plot saved to plot.png" << std::endl;
    return 0;
}

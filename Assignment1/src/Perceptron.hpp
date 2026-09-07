#ifndef PERCEPTRON_HPP
#define PERCEPTRON_HPP

#include <vector>
#include <functional>
using namespace std;


// Activation function and its mathematical derivative

struct Activation{
    function<double(double)> func;
    function<double(double)> derivative;    
};

class Perceptron{
private:
    vector<double> weights;
    double bias;
    double learning_rate;
    Activation activation;  

public:
    //Initialize weights and bias
    Perceptron(int input_size,double lr,Activation act);

    // The Forward Pass
    double predict(const vector<double>& inputs);

    // Gradient Descent step 
    void train_step(const vector<double>& inputs,double target);

    // Getters for python plots
    vector<double>getWeights() const { return weights; }
    double getBias() const { return bias;}

};


#endif
#include "Perceptron.hpp"

Perceptron::Perceptron(int input_size,double lr,Activation act){
    learning_rate = lr;
    activation = act;
    bias = 0.0;

    // Assign takes (number_of_elements,value)
    weights.assign(input_size,bias);
}

// Forward Pass (Predict) 
double Perceptron::predict(const vector<double>& inputs) {
    double z = bias;

    for(size_t i =0;i < inputs.size(); i++){
        z += inputs[i] * weights[i];
    }

    return activation.func(z);
}

// Gradient Descent 
void Perceptron::train_step(const vector<double>& inputs,double target){
    // Forward Pass
    double z = bias;

    for(size_t i =0;i < inputs.size(); i++){
        z += inputs[i] * weights[i];
    }

    // Prediction
    double prediction = activation.func(z);

    // Calculate Error
    double error = target - prediction;

    // Calculate Gradient
    double gradient = (error * activation.derivative(z));

    // Update Weights 
    for(size_t i =0;i < inputs.size();i++){
        weights[i] = weights[i] + (learning_rate * gradient * inputs[i]);
    }

    // Update Bias
    bias += (learning_rate * gradient);
}
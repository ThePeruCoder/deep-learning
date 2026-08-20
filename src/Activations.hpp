#ifndef ACTIVATIONS_HPP
#define ACTIVATIONS_HPP

#include <cmath>
#include "Perceptron.hpp"

// Linear Activation

inline double linear_func(double z){
    return z;
}

inline double linear_deriv(double z){
    return 1.0;
}

inline Activation getLinear(){
    return {linear_func,linear_deriv};
}

// Logistic / Sigmoid Activation

inline double sigmoid_func(double z){
    return 1.0 / (1.0 + exp(-z));
}

inline double sigmoid_deriv(double z){
    double fz = sigmoid_func(z);
    return fz * (1.0 - fz);
}

inline Activation getSigmoid(){
    return {sigmoid_func,sigmoid_deriv};
}


// Tanh  Activation

inline double tanh_func(double z){
    return tanh(z);
}


inline double tanh_deriv(double z){
    double fz = tanh_func(z);
    return 1.0 - (fz * fz);
}

inline Activation getTanh(){
    return {tanh_func,tanh_deriv};
}

#endif
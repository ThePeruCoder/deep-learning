#include <iostream>
#include <vector>
#include "Dataloader.hpp"
#include "Perceptron.hpp"
#include "Activations.hpp"

using namespace std;

void run_regression(){
    cout << " Running Regression (Bivariate) " <<  endl;


    vector<DataPoint> train_set;
    vector<DataPoint> test_set;

    Dataloader::loadRegressionData("Data/Regression/BivariateData/26.csv",train_set,test_set);

    int input_size = train_set[0].features.size();
    Perceptron p(input_size,0.001,getLinear());

    int epochs = 1000;

    for(int e = 0; e < epochs ; e++){
        for(int i = 0 ; i < train_set.size(); i++){
            p.train_step(train_set[i].features,train_set[i].label);
        }       
    }

    cout << " Training Complete" << endl;
    cout << " Final Weight(s): "; 
    
    for (double w : p.getWeights()){
        cout << w << " ";
    }
    cout << endl;
    cout << " Final Bias: " << p.getBias() << endl;


    // Evaluate on test

    double total_error = 0.0;

    for(int i =0 ; i < test_set.size();i++){
        double prediction = p.predict(test_set[i].features);
        double error = test_set[i].label - prediction;
        total_error += (error * error); 
    }

    double mse = total_error / test_set.size();

    cout << "Test results" << endl;
    cout << "Mean Squared Error (MSE) on test set: " << mse << endl;
}


int main(){
    run_regression();
    return 0;
}
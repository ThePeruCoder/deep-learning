#include <iostream>
#include <vector>
#include "Dataloader.hpp"
#include "Perceptron.hpp"
#include "Activations.hpp"

using namespace std;


vector<DataPoint> filter_relabel(const vector<DataPoint>& full_data,double target_class,double negative_class ){
    vector<DataPoint> filtered;

    for(size_t i = 0; i < full_data.size(); i++){
        if(full_data[i].label == target_class){
            DataPoint pt = full_data[i];
            pt.label = 1.0;
            filtered.push_back(pt);
        }else if(full_data[i].label == negative_class){
            DataPoint pt = full_data[i];
            pt.label = 0.0;
            filtered.push_back(pt);
        }
    }

    return filtered;
}


double calculate_accuracy(Perceptron& p,const vector<DataPoint>& test_data){
    int correct_guesses = 0;

    for(size_t i = 0; i < test_data.size(); i++){
        double pred = p.predict(test_data[i].features);
        double guess;
        if(pred >= 0.5){
            guess = 1.0;
        }else{
            guess = 0.0;
        }
        if(guess == test_data[i].label){
            ++correct_guesses;
        }
    }

    return (static_cast<double>(correct_guesses) / test_data.size()) * 100.0;
}


void helper(const vector<DataPoint>& train,const vector<DataPoint>& test,double target,double negative,const string& match_name){
    cout << " Match: " << match_name << endl;

    vector<DataPoint> train_subset = filter_relabel(train,target,negative);
    vector<DataPoint> test_subset = filter_relabel(test,target,negative);

    int input_size = train[0].features.size();  
    Perceptron p(input_size,0.001,getSigmoid());

    int epochs = 1000;

    for(int e = 0; e < epochs ; e++){
        for(size_t i = 0 ; i < train_subset.size(); i++){
            p.train_step(train_subset[i].features,train_subset[i].label);
        }       
    }

    double accuracy = calculate_accuracy(p, test_subset);
    cout << " Accuracy: " << accuracy << "%" << endl;
}


void run_ls_classification(){
    cout << " Linearly Separable(One vs One)" << endl;

    vector<DataPoint> train_set;
    vector<DataPoint> test_set;

    Dataloader::loadLSData(
        "Data/Classification/LS_Group26/Class1.txt",
        "Data/Classification/LS_Group26/Class2.txt",
        "Data/Classification/LS_Group26/Class3.txt",
        train_set,
        test_set
    );

    helper(train_set, test_set, 0.0, 1.0, "Class 0 vs Class 1");
    helper(train_set, test_set, 0.0, 2.0, "Class 0 vs Class 2");
    helper(train_set, test_set, 1.0, 2.0, "Class 1 vs Class 2");

}


void run_nls_classification(){
    cout << " Non Linearly Separable(One vs One)" << endl;

    vector<DataPoint> train_set;
    vector<DataPoint> test_set;

    Dataloader::loadNLSData(
        "Data/Classification/NLS_Group26.txt",
        train_set,
        test_set
    );

    helper(train_set, test_set, 0.0, 1.0, "Class 0 vs Class 1");
    helper(train_set, test_set, 0.0, 2.0, "Class 0 vs Class 2");
    helper(train_set, test_set, 1.0, 2.0, "Class 1 vs Class 2");
}


void run_bivariate(){
    cout << " Running Regression (Bivariate) " <<  endl;


    vector<DataPoint> train_set;
    vector<DataPoint> test_set;

    Dataloader::loadRegressionData("Data/Regression/BivariateData/26.csv",train_set,test_set);

    int input_size = train_set[0].features.size();
    Perceptron p(input_size,0.001,getLinear());

    int epochs = 1000;

    for(int e = 0; e < epochs ; e++){
        for(size_t i = 0 ; i < train_set.size(); i++){
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

    for(size_t i =0 ; i < test_set.size();i++){
        double prediction = p.predict(test_set[i].features);
        double error = test_set[i].label - prediction;
        total_error += (error * error); 
    }

    double mse = total_error / test_set.size();

    cout << "Test results" << endl;
    cout << "Mean Squared Error (MSE) on test set: " << mse << endl;
}

void run_univariate(){
    cout << " Running Regression (Univariate) " <<  endl;


    vector<DataPoint> train_set;
    vector<DataPoint> test_set;

    Dataloader::loadRegressionData("Data/Regression/UnivariateData/26.csv",train_set,test_set);

    int input_size = train_set[0].features.size();
    Perceptron p(input_size,0.001,getLinear());

    int epochs = 1000;

    for(int e = 0; e < epochs ; e++){
        for(size_t i = 0 ; i < train_set.size(); i++){
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

    for(size_t i =0 ; i < test_set.size();i++){
        double prediction = p.predict(test_set[i].features);
        double error = test_set[i].label - prediction;
        total_error += (error * error); 
    }

    double mse = total_error / test_set.size();

    cout << "Test results" << endl;
    cout << "Mean Squared Error (MSE) on test set: " << mse << endl;
}

int main(){
    // run_bivariate();
    // run_univariate();
    // run_ls_classification();
    run_nls_classification();
    return 0;
}
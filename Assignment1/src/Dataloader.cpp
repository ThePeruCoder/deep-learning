#include "Dataloader.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <random>
using namespace  std;

// Shuffle and Split

void Dataloader::splitData(vector<DataPoint>& fullData,vector<DataPoint>& trainData,vector<DataPoint>& testData,double trainRatio){
    //Set up the random number generator
    random_device rd;
    mt19937 g(rd());

    //Shuffle the original data in place
    shuffle(fullData.begin(),fullData.end(),g);

    // Find the exact index where 70% ends
    int splitIndex = static_cast<int>(fullData.size() * trainRatio);

    //Copy the data into the train and test vectors
    trainData.assign(fullData.begin(),fullData.begin()+splitIndex);
    testData.assign(fullData.begin()+splitIndex,fullData.end());   
}

// Linearly Seperable Helper

vector<DataPoint> Dataloader::parseClassificationFile(const string& filename,double label){
    vector<DataPoint> data;
    ifstream file(filename);


    //Safety check to ensure the file path is correct
    if(!file.is_open()){
        cerr << "Error: Could not open " << filename << endl;
        return data; //Return empty vector if it fails
    }

    double x1, x2;
    
    // The ">>" operator automatically skips spaces and newlines
    while(file >> x1 >> x2){
        DataPoint pt;
        pt.features = {x1,x2}; 
        pt.label = label;
        data.push_back(pt);
    }

    return data;
}


void Dataloader::loadLSData(const string& class1_file,const string& class2_file,const string& class3_file,
                            vector<DataPoint>& train_set,vector<DataPoint>& test_set,double train_ratio){
    

    // Read and label the individual files
    vector<DataPoint> class1 = parseClassificationFile(class1_file,0.0);
    vector<DataPoint> class2 = parseClassificationFile(class2_file,1.0);
    vector<DataPoint> class3 = parseClassificationFile(class3_file,2.0);

    // Perform 70-30 split on each class individually
    vector<DataPoint> train1, test1, train2, test2, train3, test3;
    splitData(class1 ,train1 ,test1 ,train_ratio);
    splitData(class2 ,train2 ,test2 ,train_ratio);
    splitData(class3 ,train3 ,test3 ,train_ratio);

    // Combine into master sets using vector::insert
    train_set.insert(train_set.end(),train1.begin(),train1.end());
    train_set.insert(train_set.end(),train2.begin(),train2.end());
    train_set.insert(train_set.end(),train3.begin(),train3.end());

    test_set.insert(test_set.end(),test1.begin(),test1.end());
    test_set.insert(test_set.end(),test2.begin(),test2.end());
    test_set.insert(test_set.end(),test3.begin(),test3.end());

    // Final Shuffle of the training set
    random_device rd;
    mt19937 g(rd());
    shuffle(train_set.begin(), train_set.end(), g);
    
}

void Dataloader::loadNLSData(const string& filename,vector<DataPoint>& train_set,vector<DataPoint>& test_set,double train_ratio){
    ifstream file(filename);
    if(!file.is_open()){
        cerr << "Failed to open " <<  filename <<  endl;
        return;
    }

    string header;
    // Skip the first line of text
    getline(file,header);
    
    vector<DataPoint> class1, class2, class3;
    double x1,x2;
    int count = 0;

    // Reads space-seperated coordinates
    while(file >> x1 >> x2){
        DataPoint pt;
        pt.features = {x1,x2};

        // Route the data based on the line count

        if(count < 500){
            pt.label = 0.0;
            class1.push_back(pt);
        }else if(count < 1000){
            pt.label = 1.0;
            class2.push_back(pt);
        }else{
            pt.label = 2.0;
            class3.push_back(pt);
        }
        count++;
    }

    // Perform 70-30 split on each class individually
    vector<DataPoint> train1, test1, train2, test2, train3, test3;
    splitData(class1 ,train1 ,test1 ,train_ratio);
    splitData(class2 ,train2 ,test2 ,train_ratio);
    splitData(class3 ,train3 ,test3 ,train_ratio);

    // Combine into master sets using vector::insert
    train_set.insert(train_set.end(),train1.begin(),train1.end());
    train_set.insert(train_set.end(),train2.begin(),train2.end());
    train_set.insert(train_set.end(),train3.begin(),train3.end());

    test_set.insert(test_set.end(),test1.begin(),test1.end());
    test_set.insert(test_set.end(),test2.begin(),test2.end());
    test_set.insert(test_set.end(),test3.begin(),test3.end());

    // Final Shuffle of the training set
    random_device rd;
    mt19937 g(rd());
    shuffle(train_set.begin(), train_set.end(), g);
       
}

void Dataloader::loadRegressionData(const string& filename,vector<DataPoint>& train_set,vector<DataPoint>& test_set,double train_ratio){
    vector<DataPoint> full_data;
    ifstream file(filename);


    if(!file.is_open()){
        cerr << "Error: Could not open " << filename << endl;
        return; 
    }


    string line;

    // Parses comma-seperated values dynamically for 1D or 2D data
    while(getline(file,line)){
        if(line.empty()){
            continue; // Skip trailing empty lines
        }

        stringstream ss(line);
        string value;
        vector<double> row_values;
        
        // Split the row by commas
        while(getline(ss, value ,',')){
            row_values.push_back(stod(value)); // Convert string to double
        }

        if(row_values.empty()){
            continue;
        }


        DataPoint pt;
        pt.label = row_values.back(); // Last column is target (y)
        
        // Everything before the last column is a feature

        for(size_t i = 0;i < row_values.size() - 1; ++i){
            pt.features.push_back(row_values[i]);
        }

        full_data.push_back(pt);
    }
    // Standard split for Regression
    splitData(full_data, train_set, test_set, train_ratio);

}
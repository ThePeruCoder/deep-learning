#ifndef DATALOADER_HPP
#define DATALOADER_HPP
#include <vector>
#include <string>
using namespace std;
// Strucuture to hold a single row of data
struct DataPoint{
    vector<double> features;
    double label;
};


class Dataloader{
public:
      // <<<<<<<<<<<<< Classification Datasets >>>>>>>>>>>>>>>>>>>   

      //Dataset 1 : Linearly Seperable
      static void loadLSData(const string& class1_file,const string& class2_file,const string& class3_file,
                             vector<DataPoint>& train_set,vector<DataPoint>& test_set, double train_ratio = 0.7
                            );
      
      
      //Dataset 2: Non-linearly Seperable 
      static void loadNLSData(const string& filename,vector<DataPoint>& train_set,
                              vector<DataPoint>& test_set, double train_ratio = 0.7
                            );
      
      // <<<<<<<<<<<<<<< Regression Datasets >>>>>>>>>>>>>>>>>>>>>

      //Datasets 3 & 4 : Univariate and Bivariate 
      static void loadRegressionData(const string& filename,vector<DataPoint>& train_set,
                            vector<DataPoint>& test_set, double train_ratio = 0.7
                            );

private:
        //Reads a class file, extracts coordinates, and assigns the given label to all its DataPoints.
        static vector<DataPoint> parseClassificationFile(const string& filename,double label);
        
        //Helper to shuffle and split a vector 70/30
        static void splitData(vector<DataPoint>& fullData,
                              vector<DataPoint>& trainData,
                              vector<DataPoint>& testData,
                              double trainRatio);
};

#endif

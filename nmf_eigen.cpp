//this code is NMF program with probability calculation using Eigen library
//
// the code written by @mokemoketa

#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <sstream>

//these library for checking whether string can be translated into values
#include <cctype>
#include <algorithm>

#define EIGEN_NO_DEBUG //including this to speed up the program
//#define EIGEN_DONT_PARALLELIZE
//#define EIGEN_MPL2_ONLY

#include "Eigen/Dense"
#include <random>

#define MAXITER 20 //max iteration limit

#include "exportCsv.h"

//output the matrix to the csv file
void  exportData(std::string filename,vector<MatrixXd> data)
{
  ofstream ofs(filename);
  vector<MatrixXd>::iterator it;
  for(it = data.begin(); it != data.end(); it++ )
  {
    for (int i = 0; i < (*it).rows(); i++)
    {
      ofs << (*it)(i, 0);
      if (i != (*it).rows()-1) {
      ofs << ",";
      }
    }
    ofs << endl;
  }
  ofs.close();
  return;
}

//rows=150,cols=4 for iris data

//calculate the i-divergence between matrixes X and Y
double i_div(Eigen::MatrixXd X, Eigen::MatrixXd Y)
{
  //check the dimensions of both matrix
  if (X.rows() != Y.rows() || X.cols() != Y.cols() )
  {
    std::cout << "Error1-1:Input matrices are not the same dimension!" << std::endl;
    return -1.0;
  }
  double sum = 0;
  for(int i = 0; i < X.rows(); i++){
    for(int j = 0; j < X.cols(); j++){
      sum += X(i,j) * log(X(i,j)/Y(i,j)) - X(i,j) + Y(i,j);
    }
  }
  return sum;
}

//calculate the i-divergence between matrixes X and Y with probability matrix M
double i_div(Eigen::MatrixXd X, Eigen::MatrixXd Y, Eigen::MatrixXd M)
{
  //check the dimensions of all matrixes
  if (X.rows() != Y.rows() || X.rows() != M.rows() || X.cols() != Y.cols() || X.cols() != M.cols())
  {
    std::cout << "Error1-2:Input matrices are not the same dimension!" << std::endl;
    return -1.0;
  }
  double sum = 0;
  for(int i = 0; i < X.rows(); i++){
    for(int j = 0; j < X.cols(); j++){
      if(M(i,j) && X(i,j) && Y(i,j)){
        sum += X(i,j) * log(X(i,j)/Y(i,j)) - X(i,j) + Y(i,j);
      }
    }
  }
  return sum;
}

//calculate the euclid distance
double euc_err(Eigen::MatrixXd X, Eigen::MatrixXd Y)
{
  if (X.rows() != Y.rows() || X.cols() != Y.cols() )
  {
    std::cout << "Error2-1:Input matrices are not the same dimension!" << std::endl;
    return -1.0;
  }
  return (Y - X).squaredNorm();
}
//calculate the euclid distance with probability
double euc_err(Eigen::MatrixXd X, Eigen::MatrixXd Y, Eigen::MatrixXd M)
{
  if (X.rows() != Y.rows() || X.cols() != Y.cols() )
  {
    std::cout << "Error2-2:Input matrices are not the same dimension!" << std::endl;
    return -1.0;
  }

  return (M.array()*(Y-X).array()).matrix().squaredNorm();
}

std::vector<std::string> split(const std::string &str, char sep)
{
    std::vector<std::string> v;
    std::stringstream ss(str);
    std::string buffer;
    while( std::getline(ss, buffer, sep) ) {
        v.push_back(buffer);
    }
    return v;
}

//this function originally by https://gist.github.com/infusion/43bd2aa421790d5b4582
Eigen::MatrixXd readCSV(std::string file, int rows, int cols) {

  using namespace std;

  std::ifstream in(file);
  std::string line;
  std::string token;
  std::size_t pos;

  int row = 0;
  int col = 0;

  Eigen::MatrixXd res = Eigen::MatrixXd(rows, cols);

  if (in.is_open()) {
    std::getline(in, line); //skip the first line of text
    while (std::getline(in, line)) {
      col = 0;
      vector<std::string> elems = split(line, ' '); //make a line divided by tab

      for(int i = 1; i < elems.size(); i++){
        pos = elems[i].find(".");
        token = elems[i];
        if(pos != string::npos){
          token.replace(pos, 1, "");
        }
        if(std::all_of(token.begin(), token.end(), ::isdigit)){ //lambda式が使えず
          res(row, col) = stof(elems[i]); //atofではダメだった
          col++;
          //cout << "elems[i] is:" << elems[i]  << ", token :" << token << endl;
        }
      }
      row++;
    }
    in.close();
  }
  return res;
}

//refresh the euclid distance function
void refresh_euc(Eigen::MatrixXd &X, Eigen::MatrixXd &T, Eigen::MatrixXd &V){
  Eigen::MatrixXd Y = T * V;
  T.array() = T.array() * (X * V.transpose()).array() /
                     (T * V * V.transpose()).array();
  V.array() = V.array() * (V.transpose() * X).array() /
                     (T.transpose() * T * V).array();
  /*
  for(int i = 0; i < X.rows(); i++){
    for(int k = 0; k < T.cols(); k++){
      double numer = 0;
      double denom = 0;
      for(int j = 0; j < X.cols(); j++){
        numer += X(i,j) * V(k,j);
        denom += Y(k,j) * V(i,j);
      }
      T(i,k) = T(i,k) * numer / denom;
    }
  }
  for(int k = 0; k < V.rows(); k++){
    for(int j = 0; j < X.cols(); j++){
      double numer = 0;
      double denom = 0;
      for(int i = 0; i < X.rows(); i++){
        numer += X(i,j) * T(i,k);
        denom += Y(i,k) * T(i,j);
      }
      V(k,j) = V(k,j) * numer / denom;
    }
  }
  */
  return;
}

/***This function have to be rewriten later***/
void refresh_euc(Eigen::MatrixXd &X, Eigen::MatrixXd &T, Eigen::MatrixXd &V, Eigen::MatrixXd &M){
  Eigen::MatrixXd Y = T * V;
  cout << "This function, \"refresh_euc(X,T,V,M)\" is still under construction!" << endl;
  T.array() = T.array() * (X * V.transpose()).array() /
                     (T * V * V.transpose()).array();
  V.array() = V.array() * (V.transpose() * X).array() /
                     (T.transpose() * T * V).array();
  /*
  for(int i = 0; i < X.rows(); i++){
    for(int k = 0; k < T.cols(); k++){
      double numer = 0;
      double denom = 0;
      for(int j = 0; j < X.cols(); j++){
        numer += X(i,j) * V(k,j);
        denom += Y(k,j) * V(i,j);
      }
      T(i,k) = T(i,k) * numer / denom;
    }
  }
  for(int k = 0; k < V.rows(); k++){
    for(int j = 0; j < X.cols(); j++){
      double numer = 0;
      double denom = 0;
      for(int i = 0; i < X.rows(); i++){
        numer += X(i,j) * T(i,k);
        denom += Y(i,k) * T(i,j);
      }
      V(k,j) = V(k,j) * numer / denom;
    }
  }
  */
  return;
}

//refresh the i-divergence function
void refresh_i(Eigen::MatrixXd &X, Eigen::MatrixXd &T, Eigen::MatrixXd &V){
  Eigen::MatrixXd Y = T * V;

  for(int i = 0; i < X.rows(); i++){
    for(int k = 0; k < T.cols(); k++){
      double numer = 0;
      double denom = 0;
      for(int j = 0; j < X.cols(); j++){
        if(Y(i,j) != 0.0){
          numer += X(i,j) / Y(i,j) * V(k,j);
          denom += V(k,j);
        }
      }
      if (denom != 0.0) {
        T(i,k) = T(i,k) * numer / denom;
      }else{
        T(i,k) = 0.0;
      }
    }
  }
  Y = T * V;
  for(int k = 0; k < V.rows(); k++){
    for(int j = 0; j < X.cols(); j++){
      double numer = 0;
      double denom = 0;
      for(int i = 0; i < X.rows(); i++){
        if(Y(i,j) != 0.0){
          numer += X(i,j) / Y(i,j) * T(i,k);
          denom += T(i,k);
        }
      }
      V(k,j) = V(k,j) * numer / denom;
    }
  }
  return;
}
//refresh the i-divergence function with probability
void refresh_i(Eigen::MatrixXd &X, Eigen::MatrixXd &T, Eigen::MatrixXd &V, Eigen::MatrixXd &M){
  Eigen::MatrixXd Y = T * V;

  for(int i = 0; i < X.rows(); i++){
    for(int k = 0; k < T.cols(); k++){
      double numer = 0;
      double denom = 0;
      for(int j = 0; j < X.cols(); j++){
        if(Y(i,j) != 0.0){
          numer += M(i,j)*X(i,j) / Y(i,j) * V(k,j);
          denom += M(i,j)*V(k,j);
        }
      }
      if (denom != 0.0) {
        T(i,k) = T(i,k) * numer / denom;
      }else{
        T(i,k) = 0.0;
      }
    }
  }
  Y = T * V;
  for(int k = 0; k < V.rows(); k++){
    for(int j = 0; j < X.cols(); j++){
      double numer = 0;
      double denom = 0;
      for(int i = 0; i < X.rows(); i++){
        if(Y(i,j) != 0.0){
          numer += M(i,j)*X(i,j) / Y(i,j) * T(i,k);
          denom += M(i,j)*T(i,k);
        }
      }
      if (denom != 0.0) {
        V(k,j) = V(k,j) * numer / denom;
      }else{
        V(k,j) = 0.0;
      }
    }
  }
  return;
}

int main(int argc, char* argv[]){
  using namespace Eigen;
  using namespace std;

  if(argc != 5){
    std::cout << "Usage:" << argv[0] << " [inputfile] [rows] [columns] [dimensions]" << std::endl;
    return -1;
  }

  MatrixXd X1 = readCSV(argv[1], atoi(argv[2]), atoi(argv[3])); //this sentence makes values in X1

  //for randomization with mercen function
  std::random_device rnd;
  std::mt19937 mt(rnd());
  std::uniform_real_distribution<> rand01(0.0, 1.0);

  int k = atoi(argv[4]);

  //the initiation of T,V
  MatrixXd T1 = MatrixXd(X1.rows(),k);
  MatrixXd V1 = MatrixXd(k,X1.cols());
  //the initialization of T,V with rand values [0.0,1.0]
  for (int i = 0; i < X1.rows(); i++){
    for (int j = 0; j < k; j++){
      T1(i,j) = rand01(mt);
    }
  }
  for (int i = 0; i < X1.cols(); i++){
    for (int j = 0; j < k; j++){
      V1(j,i) = rand01(mt);
    }
  }
  //save the csv files for checking the change of div on each iteration
  ofstream ofs("iter.csv");
  //ofs << "0," << i_div(X1,T1*V1, M1) << endl;
  //ofs1 << "0," << i_div(X1,T1*V1, M2) << endl;
  ofs << "0," << i_div(X1,T1*V1) << endl;

  for(int i = 1; i <= MAXITER; i++){
    refresh_i(X1, T1, V1); //refresh function for i-divergence
    //refresh_euc(X1, T1, V1, M1); //refresh function for euclid error

    /*iteration check for euclid error*/
    //ofs << i << "," << euc_err(X1, T1*V1, M1) << endl;
    //ofs1 << i << "," << euc_err(X1, T1*V1, M2) << endl;

    /*iteration check for i-divergence*/
    double now_div = i_div(X1,T1*V1);
    ofs << i << "," << now_div << endl;
    if(now_div == 0.0){
      cout << "divergence is now 0. We will stop iteration." << endl;
      break;
    }
  }

  ofstream ofs2("dataT1.csv");
  for(int i=0; i < T1.rows(); i++){
    for(int j=0; j < T1.cols()-1; j++){
      ofs2 << T1(i,j) << ",";
    }
    ofs2 << T1(i,T1.cols()-1) << endl;
  }
  ofstream ofs3("dataV1.csv");
  for(int i=0; i < V1.rows(); i++){
    for(int j=0; j < V1.cols()-1; j++){
      ofs3 << V1(i,j) << ",";
    }
    ofs3 << V1(i,V1.cols()-1) << endl;
  }

  return 0;
}

#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <sstream>

//these library for checking whether string can be translated into values
#include <cctype>
#include <algorithm>

//#define EIGEN_NO_DEBUG
//#define EIGEN_DONT_PARALLELIZE
//#define EIGEN_MPL2_ONLY

#include "Eigen/Core"
#include "unsupported/Eigen/CXX11/Tensor"
#include <random>
#include "exportCsv.h"

#define CONST1 0

#define MAXITER 20
//rows=150,cols=4 for iris data

void  exportData(std::string filename,Eigen::Tensor<double, 3> data)
{
 ofstream ofs(filename);

 for(int i = 0; i < data.dimension(0); i++){
   for (int j = 0; j < data.dimension(1); j++){
     for (int k = 0; k < data.dimension(2); k++){
       ofs << data(i, j, k);
       if (k != data.dimension(2)-1) {
         ofs << ",";
       }
     }
     ofs << endl;
   }
 }
 ofs.close();

 return;
}

Eigen::MatrixXd mode_unfold(Eigen::Tensor<double, 3> X, int mode){
  if(mode < 0 || mode > 3){
    cout << "Error: in mode_unfold, mode is incorrect value!" << endl;
  }
  int dim = 1;
  int is[2];
  int cnt = 0;
  for(int i = 0; i < 3; i++){
    if(i != mode-1){
      is[cnt++] = i;
      dim *= X.dimension(i);
    }
  }
  int js[3];
  int tmp = 1;
  for(int k = 1; k <= 3; k++){
    tmp = 1;
    for(int m = 1; m <= k-1; m++){
      if(m != mode) tmp *= X.dimension(m-1);
    }
    js[k-1] = tmp;
  }
  int at[3];
  Eigen::MatrixXd Xf = Eigen::MatrixXd(X.dimension(mode-1),dim);
  for(int i = 0; i < X.dimension(mode-1); i++){
    for(int j = 1; j <= X.dimension(is[0]); j++){
      for(int k = 1; k <= X.dimension(is[1]); k++){
        at[is[0]] = j-1;
        at[is[1]] = k-1;
        at[mode-1] = i;
        Xf(i,(j-1)*js[is[0]]+(k-1)*js[is[1]]) = X(at[0],at[1],at[2]);
      }
    }
  }
  return Xf;
}

Eigen::MatrixXd kr_cross(Eigen::MatrixXd A, Eigen::MatrixXd B)
{
  if(A.cols() != B.cols()){
    cout << "Error: in kr_cross, # of dims are different!" << endl;
    //return MatrixXd::Ones(1,1);
  }
  Eigen::MatrixXd X = Eigen::MatrixXd(A.rows()*B.rows(),A.cols());
  for(int i = 0; i < A.rows(); i++){
    for(int j = 0; j < B.rows(); j++){
      for(int k = 0; k < A.cols(); k++){
        X(i*B.rows()+j,k) = A(i,k)*B(j,k);
      }
    }
  }
  return X;
}

//calculate the kl-divergence
double kl_div(Eigen::Tensor<double, 3> X, Eigen::Tensor<double, 3> Y)
{
  // //check the dimensions of both matrix
  // if (X.rows() != Y.rows() || X.cols() != Y.cols() )
  // {
  //   std::cout << "Input matrixes are not the same dimension!" << std::endl;
  //   return -1.0;
  // }
  double sum = 0;
  for(int n = 0; n < X.dimension(0); n++){
    for(int i = 0; i < X.dimension(1); i++){
      for(int j = 0; j < X.dimension(2); j++){
        if(X(n,i,j) != 0.0){
          sum += - X(n,i,j) * log(Y(n,i,j)) + Y(n,i,j);
        }else{
          sum += Y(n,i,j);
        }
      }
    }
  }
  return sum;
}
//calculate the kl-divergence with probability
double kl_div(Eigen::Tensor<double, 3> X, Eigen::MatrixXd U, Eigen::MatrixXd T, Eigen::MatrixXd V, Eigen::Tensor<double, 3> M)
{
  // //check the dimensions of both matrix
  // if (X.rows() != Y.rows() || X.rows() != M.rows() || X.cols() != Y.cols() || X.cols() != M.cols())
  // {
  //   std::cout << "Input matrixes are not the same dimension!" << std::endl;
  //   return -1.0;
  // }
  double sum = 0;
  for(int n = 0; n < X.dimension(0); n++){
    for(int i = 0; i < X.dimension(1); i++){
      for(int j = 0; j < X.dimension(2); j++){
        double y_sum = 0;
        for(int r = 0; r < U.cols(); r++){
          y_sum += U(n,r)*T(i,r)*V(j,r);
        }
        if(M(n,i,j) && X(n,i,j) && y_sum){
          if(X(n,i,j) != 0.0){
            sum += - X(n,i,j) * log(y_sum) + y_sum + CONST1;
          }else{
            sum += y_sum + CONST1;
          }
        }
      }
    }
  }
  return sum;
}

//calculate the i-divergence
double i_div(Eigen::MatrixXd X, Eigen::MatrixXd Y)
{
  //check the dimensions of both matrix
  if (X.rows() != Y.rows() || X.cols() != Y.cols() )
  {
    std::cout << "Input matrixes are not the same dimension!" << std::endl;
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
//calculate the i-divergence with probability
double i_div(Eigen::MatrixXd X, Eigen::MatrixXd Y, Eigen::MatrixXd M)
{
  //check the dimensions of both matrix
  if (X.rows() != Y.rows() || X.rows() != M.rows() || X.cols() != Y.cols() || X.cols() != M.cols())
  {
    std::cout << "Input matrixes are not the same dimension!" << std::endl;
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

//calculate the euclid distance with probability
double euc_err(Eigen::Tensor<double,3> X, Eigen::Tensor<double,3> Y)
{
  double sum = 0;
  for(int n = 0; n < X.dimension(0); n++){
    for(int i = 0; i < X.dimension(1); i++){
      for(int j = 0; j < X.dimension(2); j++){
        sum += pow(X(n,i,j) - Y(n,i,j),2);
      }
    }
  }
  return sum;
}
//calculate the euclid distance
double euc_err(Eigen::Tensor<double,3> X, Eigen::Tensor<double,3> Y, Eigen::Tensor<double,3> M)
{
  double sum = 0;
  for(int n = 0; n < X.dimension(0); n++){
    for(int i = 0; i < X.dimension(1); i++){
      for(int j = 0; j < X.dimension(2); j++){
        if(M(n,i,j)){
          sum += pow(X(n,i,j) - Y(n,i,j),2);
        }
      }
    }
  }
  return sum;
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

Eigen::MatrixXd krc_vec(Eigen::MatrixXd A, Eigen::MatrixXd B) {

  const int asize = A.rows();
  const int bsize = B.rows();
  const int K = A.cols();

  Eigen::MatrixXd X = Eigen::MatrixXd(asize*bsize,K);

  for(int i = 1; i <= asize; i++){
    for(int j = 1; j <= bsize; j++){
      for(int k=0; k < K; k++){
        X(i*j-1,k) = A(i,k) * B(j,k);
      }
    }
  }
  return X;
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
    std::getline(in, line);
    while (std::getline(in, line)) {
      col = 0;
      vector<std::string> elems = split(line, ',');

      for(int i = 0; i < elems.size(); i++){
        pos = elems[i].find(".");
        token = elems[i];
        if(pos != string::npos){
          token.replace(pos, 1, "");
        }
        if(std::all_of(token.begin(), token.end(), ::isdigit)){ //lambda式が使えなかった
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

Eigen::Tensor<double,3> readCSV(std::string file, int rows, int cols, int deps) {

  using namespace std;

  std::ifstream in(file);
  std::string line;
  std::string token;
  std::size_t pos;

  int row = 0;
  int col = 0;
  int dep = 0;

  Eigen::Tensor<double, 3> res(rows, cols, deps);

  if (in.is_open()) {
    std::getline(in, line);
    while (std::getline(in, line)) {
      col = 0;
      vector<std::string> elems = split(line, ',');

      for(int i = 0; i < elems.size(); i++){
        pos = elems[i].find(".");
        token = elems[i];
        if(pos != string::npos){
          token.replace(pos, 1, "");
        }
        if(std::all_of(token.begin(), token.end(), ::isdigit)){ //lambda式が使えなかった
          res(row, col, dep) = stof(elems[i]); //atofではダメだった
          if(dep == deps-1){
            col++;
            dep = 0;
          }else{
            dep++;
          }
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
        numer += X(i,j) / Y(i,j) * V(k,j);
        denom += V(k,j);
      }
      T(i,k) = T(i,k) * numer / denom;
    }
  }
  Y = T * V;
  for(int k = 0; k < V.rows(); k++){
    for(int j = 0; j < X.cols(); j++){
      double numer = 0;
      double denom = 0;
      for(int i = 0; i < X.rows(); i++){
        numer += X(i,j) / Y(i,j) * T(i,k);
        denom += T(i,k);
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
          numer += M(i,j)*X(i,j) / (Y(i,j) * T(i,k)+pow(10,-10));
          denom += M(i,j)*T(i,k);
        }
      }
      if (denom != 0.0) {
        V(k,j) = V(k,j) * numer / (denom + pow(10,-10));
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


  if(argc != 6){
    std::cout << "Usage:" << argv[0] << " [inputfile] [rows] [columns] [depths] [dimensions]" << std::endl;
    return -1;
  }

  /*
  Tensor<double, 3> X1(3, 4, 2);
                X1.setValues({{{1.0f,13.0f},
                              {4.0f, 16.0f},
                              {7.0f, 19.0f},
                              {10.0f, 22.0f}},
                              {{2.0f, 14.0f},
                              {5.0f, 17.0f},
                              {8.0f, 20.0f},
                              {11.0f, 23.0f}},
                              {{3.0f, 15.0f},
                              {6.0f, 18.0f},
                              {9.0f, 21.0f},
                              {12.0f, 24.0f}}});
  */
  Eigen::Tensor<double, 3> X1 = readCSV(argv[1], atoi(argv[2]), atoi(argv[3]), atoi(argv[4])); //this sentence makes values in X1

  //for randomization
  std::random_device rnd;
  std::mt19937 mt(rnd());
  std::uniform_real_distribution<> rand01(0.0, 1.0);

  Tensor<double,3> M1(X1.dimension(0),X1.dimension(1),X1.dimension(2));
  for (int i = 0; i < X1.dimension(0); i++){
    for (int j = 0; j < X1.dimension(1); j++){
      for (int k = 0; k < X1.dimension(2); k++){
        //M1(i,j,k) = (rand01(mt)>=0.5) ? 1.0:0.0;
        M1(i,j,k) = (X1(i,j,k) > 0.0) ? 1.0:0.0;
      }
    }
  }


  Tensor<double, 3> M2(X1.dimension(0),X1.dimension(1),X1.dimension(2));
  M2.setConstant(1.0f);
  M2 -= M1;
  //cout << M1 << endl;
  int r = atoi(argv[5]);







  //the initiation of U,T,V : X1 = UxTxV
  MatrixXd U = MatrixXd(X1.dimension(0),r);
  MatrixXd T = MatrixXd(X1.dimension(1),r);
  MatrixXd V = MatrixXd(X1.dimension(2),r);
  MatrixXd A1 = mode_unfold(X1, 1);
  MatrixXd A2 = mode_unfold(X1, 2);
  MatrixXd A3 = mode_unfold(X1, 3);
  MatrixXd M1_1 = mode_unfold(M1, 1);
  MatrixXd M1_2 = mode_unfold(M1, 2);
  MatrixXd M1_3 = mode_unfold(M1, 3);

  //the initialization of U,T,V with rand values [0.0,1.0]
  for (int i = 0; i < U.rows(); i++){
    for (int j = 0; j < U.cols(); j++){
      U(i,j) = rand01(mt);
    }
  }
  for (int i = 0; i < T.rows(); i++){
    for (int j = 0; j < T.cols(); j++){
      T(i,j) = rand01(mt);
    }
  }
  for (int i = 0; i < V.cols(); i++){
    for (int j = 0; j < V.cols(); j++){
      V(i,j) = rand01(mt);
    }
  }

  //for checking the change of div on each iteration
  ofstream ofs("iter1.csv");
  ofstream ofs1("iter2.csv");
  ofs << "0," << kl_div(X1,U,T,V, M1) << endl;
  ofs1 << "0," << kl_div(X1,U,T,V, M2) << endl;
  for(int i = 1; i < MAXITER; i++){
    auto Y1 = kr_cross(V,T);
    Y1.transposeInPlace();
    refresh_i(A1,U,Y1,M1_1);
    auto Y2 = kr_cross(V,U);
    Y2.transposeInPlace();
    refresh_i(A2,T,Y2,M1_2);
    auto Y3 = kr_cross(T,U);
    Y3.transposeInPlace();
    refresh_i(A3,V,Y3,M1_3);
    ofs << i << "," << kl_div(X1,U,T,V, M1) << endl;
    ofs1 << i << "," << kl_div(X1,U,T,V, M2) << endl;
  }

  ofstream ofs2_1("dataU.csv");
  // for(int i=0; i < U.rows(); i++){
  //   for(int j=0; j < U.cols()-1; j++){
  //     ofs2_1 << U(i,j) << ",";
  //   }
  //   ofs2_1 << U(i,U.cols()-1) << endl;
  // }
  ofs2_1 << U << endl;
  ofstream ofs2_2("dataT.csv");
  ofs2_2 << T << endl;
  // for(int i=0; i < T.rows(); i++){
  //   for(int j=0; j < T.cols()-1; j++){
  //     ofs2_2 << T(i,j) << ",";
  //   }
  //   ofs2_2 << T(i,T.cols()-1) << endl;
  // }
  ofstream ofs2_3("dataV.csv");
  // for(int i=0; i < V.rows(); i++){
  //   for(int j=0; j < V.cols()-1; j++){
  //     ofs2_3 << V(i,j) << ",";
  //   }
  //   ofs2_3 << V(i,V.cols()-1) << endl;
  // }
  ofs2_3 << V << endl;
  ofstream ofs2_4("dataY1.csv");
  auto Y_e = kr_cross(V,T);
  Y_e.transposeInPlace();
  ofs2_4 << U*Y_e << endl;
  ofstream ofs4("dataM_1.csv");
  // for(int i=0; i < M1_1.rows(); i++){
  //   for(int j=0; j < M1_1.cols()-1; j++){
  //     ofs4 << M1_1(i,j) << ",";
  //   }
  //   ofs4 << M1_1(i,M1_1.cols()-1) << endl;
  // }
  ofs4 << M1_1 << endl;
  return 0;
}

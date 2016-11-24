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

#include "Eigen/Dense"
#include "unsupported/Eigen/CXX11/Tensor"
#include <random>

#include "exportCsv.h"

void  exportData(std::string filename,Eigen::Tensor<double, 3> data)
{
 ofstream ofs(filename);

 Eigen::Tensor<double, 1> t(3);
 int i,j,k;
 const Eigen::Tensor<double,>

 for(i = 0; i != ; it++ )
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

int main(int argc, char* argv[]){

  Eigen::Tensor<double,3> epsilon(3,3,3);


}

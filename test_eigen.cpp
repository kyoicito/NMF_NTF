#include <iostream>
#include <string>
#include <sstream>

#define EIGEN_NO_DEBUG
#define EIGEN_DONT_PARALLELIZE
#define EIGEN_MPL2_ONLY

#include "Eigen/Core"

int main(int argc, *char argv[]){
  using namespace Eigen;
  using namespace std;





  MatrixXd X(4,5);

  X<< 1, 1, 2, 3, 1,
    0, 1, 0, 1, 1,
    2, 0, 4, 4, 0,
    3, 0, 6, 6, 0;

  cout<<X<<endl;

  MatrixXd T(4,2),V(2,5);

  T<< 1, 1,
    0, 1,
    2, 0,
    3, 0;

  V<<1, 0, 2, 2, 0,
    0, 1, 0, 1, 1;

  cout<<T*V<<endl;


  return 0;
}

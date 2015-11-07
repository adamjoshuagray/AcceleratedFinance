//
// This runs the testing logic.
//

#include "../CUDA/AFCUDA.h"
#define BOOST_TEST_MODULE CUDATest
#include <boost/test/included/unit_test.hpp>

#include <math.h>

#define EPSILON 0.000001

using namespace std;

BOOST_AUTO_TEST_CASE( MathExtensions ) {
  // Make sure that the power function makes sense.
  BOOST_CHECK(af_powf(10,0) == 1.);
  BOOST_CHECK(af_powf(10,1) == 10.);
  BOOST_CHECK(af_powf(10,2) == 100.);
  BOOST_CHECK(abs(af_powf(10,-1) - 0.1) < EPSILON);
  // Make sure that the norm pdf function makes sense.
  BOOST_CHECK(abs(af_normpdff(0) - 1. / sqrt(2. * M_PI)) < EPSILON);
  BOOST_CHECK(af_normpdff(-100) == 0);
  BOOST_CHECK(af_normpdff(100) == 0);
}

BOOST_AUTO_TEST_CASE( Curve ) {
  
}

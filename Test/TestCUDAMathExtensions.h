
#ifndef AF_TEST_CUDA_MATH_EXTENSIONS_H
#define AF_TEST_CUDA_MATH_EXTENSIONS_H

using namespace std;

//
// This makes sure that the af_powf function returns the correct values.
//
BOOST_AUTO_TEST_CASE( MathExtensions_af_powf ) {
  BOOST_CHECK(af_powf(10,0) == 1.);
  BOOST_CHECK(af_powf(10,1) == 10.);
  BOOST_CHECK(af_powf(10,2) == 100.);
  BOOST_CHECK(abs(af_powf(10,-1) - 0.1) < EPSILON);
}

//
// This makes sure that the af_normpdff function returns the correct values.
//
BOOST_AUTO_TEST_CASE( MathExtensions_af_normpdff ) {
  BOOST_CHECK(abs(af_normpdff(0) - 1. / sqrt(2. * M_PI)) < EPSILON);
  BOOST_CHECK(af_normpdff(-100) == 0);
  BOOST_CHECK(af_normpdff(100) == 0);
}

#endif // AF_TEST_CUDA_MATH_EXTENSIONS_H

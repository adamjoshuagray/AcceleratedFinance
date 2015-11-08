#ifndef AF_TEST_CUDA_RESULT_H
#define AF_TEST_CUDA_RESULT_H

using namespace std;

//
// Checks that the functions which check whether a float
// result is "unknown" works.
//
BOOST_AUTO_TEST_CASE( Result_af_ResultIsUnknownFloat ) {
  BOOST_CHECK(af_ResultIsUnknownFloat(AF_UNKNOWN_FLOAT));
  BOOST_CHECK(!af_ResultIsUnknownFloat(0));
}

//
// Checks that the functions which check whether a time_t
// result is "unknown" works.
//
BOOST_AUTO_TEST_CASE( Result_af_ResultIsUnknownTime ) {
  BOOST_CHECK(af_ResultIsUnknownTime(AF_UNKNOWN_TIME));
  BOOST_CHECK(af_ResultIsUnknownTime(0));
  BOOST_CHECK(!af_ResultIsUnknownTime(1));
}

#endif // AF_TEST_CUDA_RESULT_H

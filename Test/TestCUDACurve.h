#ifndef AF_TEST_CUDA_CURVE_H
#define AF_TEST_CUDA_CURVE_H

using namespace std;

//
// This creates a curve with enough points
// that we can use it for testing and asserting
// basic behaviour.
//
afTimeCurve_t* afTest_CreateTimeCurve() {
  int count             = 10;
  afTimeCurve_t* curve  = (afTimeCurve_t*) malloc(sizeof(afTimeCurve_t));
  curve->pairs          = (afTimeCurvePair_t*) malloc(sizeof(afTimeCurvePair_t) * count);
  curve->count          = count;
  for (int i = 0; i < count; i++) {
    curve->pairs[i].time  = (time_t) (i + 1) * 10000;
    curve->pairs[i].value = i * 10000 % 5559;
  }
  return curve;
}

//
// Tests that the time returned is the first time point in the curve.
//
BOOST_AUTO_TEST_CASE( Curve_af_TimeCurveFirstTime ) {
  afTimeCurve_t* curve  = afTest_CreateTimeCurve();
  time_t first_time     = af_TimeCurveFirstTime(curve);
  BOOST_CHECK(first_time == (time_t) 10000);
  af_TimeCurveDelete(curve);
}

//
// Tests that the time returned is the last time point in the curve.
//
BOOST_AUTO_TEST_CASE( Curve_af_TimeCurveLastTime ) {
  afTimeCurve_t* curve  = afTest_CreateTimeCurve();
  time_t first_time     = af_TimeCurveLastTime(curve);
  BOOST_CHECK(first_time == (time_t) 100000);
  af_TimeCurveDelete(curve);
}

//
// Tests the "previous" interpolation methods.
//
BOOST_AUTO_TEST_CASE( Curve_af_TimeCurveInterpolatePrevious ) {
  afTimeCurve_t* curve  = afTest_CreateTimeCurve();
  BOOST_CHECK(af_ResultIsUnknownFloat(af_TimeCurveInterpolatePrevious(curve, 0)));
  BOOST_CHECK(af_ResultIsUnknownFloat(af_TimeCurveInterpolatePrevious(curve, 10)));
  // Note that because we are using "previous" interpolation we can actually
  // extrapolate beyond the last point.
  for (int i = 0; i < 10; i++) {
    BOOST_CHECK(af_TimeCurveInterpolatePrevious(curve, 10000 * (i + 1)) == i * 10000 % 5559);
    BOOST_CHECK(af_TimeCurveInterpolatePrevious(curve, 10000 * (i + 1) + 999) == i * 10000 % 5559);
  }
  af_TimeCurveDelete(curve);
}

//
// Tests the "linear" interpolation methods.
//
BOOST_AUTO_TEST_CASE( Curve_af_TimeCurveInterpolateLinear ) {
  afTimeCurve_t* curve  = afTest_CreateTimeCurve();
  BOOST_CHECK(af_ResultIsUnknownFloat(af_TimeCurveInterpolateLinear(curve, 0)));
  BOOST_CHECK(af_ResultIsUnknownFloat(af_TimeCurveInterpolateLinear(curve, 10)));
  // Note that because we are using "linear" interpolation we can't go beyond
  // the last point in the curve.
  for (int i = 0; i < 9; i++) {
    BOOST_CHECK(af_TimeCurveInterpolateLinear(curve, 10000 * (i + 1)) == i * 10000 % 5559);
    float value = ((10000-1234) * (i * 10000 % 5559) + (1234) * ((i + 1) * 10000 % 5559)) / 10000.;
    BOOST_CHECK(abs(af_TimeCurveInterpolateLinear(curve, 10000 * (i + 1) + 1234) - value) < EPSILON);
  }
  // Make sure we get an unknown result outside the interpolation region.
  BOOST_CHECK(af_ResultIsUnknownFloat(af_TimeCurveInterpolateLinear(curve, 10000 * 10 + 1234)));
  af_TimeCurveDelete(curve);
}

//
// Tests the multiplication of curve values.
//
BOOST_AUTO_TEST_CASE( Curve_af_TimeCurveMult ) {
  afTimeCurve_t* curve      = afTest_CreateTimeCurve();
  afTimeCurve_t* new_curve  = af_TimeCurveMult(13.5, curve);
  for (int i = 0; i < 10; i++) {
    BOOST_CHECK(abs(curve->pairs[i].value * 13.5 - new_curve->pairs[i].value) < EPSILON);
  }
  af_TimeCurveDelete(curve);
  af_TimeCurveDelete(new_curve);
}

//
// Tests the inplace multiplication of curves.
//
BOOST_AUTO_TEST_CASE( Curve_af_TimeCurveMultInPlace ) {
  afTimeCurve_t* curve      = afTest_CreateTimeCurve();
  afTimeCurve_t* new_curve  = af_TimeCurveMult(1, curve);
  af_TimeCurveMultInPlace(-3.5, curve);
  for (int i = 0; i < 10; i++) {
    BOOST_CHECK(abs(curve->pairs[i].value - new_curve->pairs[i].value * -3.5) < EPSILON);
  }
  af_TimeCurveDelete(curve);
  af_TimeCurveDelete(new_curve);
}

#endif

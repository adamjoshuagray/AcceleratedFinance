
#ifndef AF_TEST_CUDA_BS_OPTION_VALUATION_H
#define AF_TEST_CUDA_BS_OPTION_VALUATION_H

//
// This creates an option for testing with.
// Any option created by this method should pass
// af_BSOptionValidate unless modified latter.
//
// type     - The type of the option - Put / Call.
// S        - The price of the underlying stock.
// K        - The strike price of the option.
// tau      - The time to expiry of the option.
// r        - The discount rate to use for the option.
// sigma    - The volatility of the underlying stock.
// q        - The dividend yield of the stock.
//
afOptionInfo_t* afTest_CreateOption(afOptionType_t type, float S, float K, float tau, float r, float sigma, float q) {
  afOptionInfo_t* option  = (afOptionInfo_t*) malloc(sizeof(afOptionInfo_t));
  option->style           = AF_OPTION_STYLE_EUROPEAN;
  option->sigma_style     = AF_OPTION_SIGMA_STYLE_SCALAR;
  option->r_style         = AF_OPTION_YIELD_STYLE_SCALAR;
  option->q_style         = AF_OPTION_DIVIDEND_STYLE_SCALAR;
  option->price           = AF_UNKNOWN_FLOAT;
  option->r_curve         = NULL;
  option->q_curve         = NULL;
  option->sigma_curve     = NULL;
  option->type            = type;
  option->S               = S;
  option->K               = K;
  option->r               = r;
  option->q               = q;
  option->tau             = tau;
  return option;
}

//
// Checks that the validation function works as expected.
//
BOOST_AUTO_TEST_CASE( BSOptionValuation_af_BSOptionValidate ) {
  afOptionInfo_t* option  = afTest_CreateOption(AF_OPTION_TYPE_CALL, 100., 100., 10. / 250., 0.05, 0.2, 0.1);
  BOOST_CHECK(af_BSOptionValidate(option));
  option->style           = AF_OPTION_STYLE_AMERICAN;
  BOOST_CHECK(!af_BSOptionValidate(option));
  option->style           = AF_OPTION_STYLE_EUROPEAN;
  option->sigma_style     = AF_OPTION_SIGMA_STYLE_CURVE;
  BOOST_CHECK(!af_BSOptionValidate(option));
  option->sigma_style     = AF_OPTION_SIGMA_STYLE_SCALAR;
  option->r_style         = AF_OPTION_YIELD_STYLE_CURVE;
  BOOST_CHECK(!af_BSOptionValidate(option));
  option->r_style         = AF_OPTION_YIELD_STYLE_SCALAR;
  option->q_style         = AF_OPTION_DIVIDEND_STYLE_CURVE;
  BOOST_CHECK(!af_BSOptionValidate(option));
  option->q_style         = AF_OPTION_DIVIDEND_STYLE_SCALAR;
  BOOST_CHECK(af_BSOptionValidate(option));
  af_OptionInfoDelete(option);
}

//
// Checks that d_1 is calculated correctly.
//
BOOST_AUTO_TEST_CASE( BSOptionValuation_af_BSOptionD_1 ) {
  BOOST_CHECK(abs(af_BSOptionD_1(10, 0.01, 0.2, 0.1, 0.1, 0.05) - 5.035) < EPSILON);
}

//
// Checks that d_2 is calculated correctly.
//
BOOST_AUTO_TEST_CASE( BSOptionValuation_af_BSOptionD_2 ) {
  BOOST_CHECK(abs(af_BSOptionD_2(10, 0.01, 0.2, 0.1, 0.1, 0.05) - 5.015) < EPSILON);
}

//
// Checks that the option price is calculated correctly.
//
//
// Checks that d_2 is calculated correctly.
//
BOOST_AUTO_TEST_CASE( BSOptionValuation_af_BSOptionPrice ) {
  afOptionInfo_t* option  = afTest_CreateOption(AF_OPTION_TYPE_CALL, 100., 100., 10. / 250., 0.05, 0.2, 0.);
  printf("px: %f\n", af_BSOptionPrice(option));
  BOOST_CHECK(abs(af_BSOptionPrice(option) - 1.449) < EPSILON);
  af_OptionInfoDelete(option);
}

#endif // AF_TEST_CUDA_BS_OPTION_VALUATION_H

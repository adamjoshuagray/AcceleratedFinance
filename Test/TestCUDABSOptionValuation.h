
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
  option->sigma           = sigma;
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
BOOST_AUTO_TEST_CASE( BSOptionValuation_af_BSOptionPrice ) {
  afOptionInfo_t* option  = afTest_CreateOption(AF_OPTION_TYPE_CALL, 100., 100., 10. / 250., 0.05, 0.2, 0.);
  BOOST_CHECK(abs(af_BSOptionPrice(option) - 1.69596) < EPSILON);
  option->type            = AF_OPTION_TYPE_PUT;
  BOOST_CHECK(abs(af_BSOptionPrice(option) - 1.49616) < EPSILON);
  option->sigma           = 0;
  BOOST_CHECK(abs(af_BSOptionPrice(option) - 0) < EPSILON);
  option->type            = AF_OPTION_TYPE_CALL;
  BOOST_CHECK(abs(af_BSOptionPrice(option) - 0.1998) < EPSILON);
  option->sigma           = 1;
  option->r               = 0.3;
  option->K               = 105;
  BOOST_CHECK(abs(af_BSOptionPrice(option) - 6.3775) < EPSILON);
  option->type            = AF_OPTION_TYPE_PUT;
  BOOST_CHECK(abs(af_BSOptionPrice(option) - 10.1250) < EPSILON);
  option->r               = 0;
  BOOST_CHECK(abs(af_BSOptionPrice(option) - 10.9056) < EPSILON);
  option->type            = AF_OPTION_TYPE_CALL;
  BOOST_CHECK(abs(af_BSOptionPrice(option) - 5.9056) < EPSILON);
  // TODO: Add dividend testing.
  option->style           = AF_OPTION_STYLE_AMERICAN;
  BOOST_CHECK(af_ResultIsUnknownFloat(af_BSOptionPrice(option)));
  af_OptionInfoDelete(option);
}

//
// Checks that the option delta is calculated correctly.
//
BOOST_AUTO_TEST_CASE( BSOptionValuation_af_BSOptionDelta ) {
  afOptionInfo_t* option  = afTest_CreateOption(AF_OPTION_TYPE_CALL, 100., 100., 10. / 250., 0.05, 0.2, 0.);
  BOOST_CHECK(abs(af_BSOptionDelta(option) - 0.5279) < EPSILON);
  option->type            = AF_OPTION_TYPE_PUT;
  BOOST_CHECK(abs(af_BSOptionDelta(option) + 0.4721) < EPSILON);
  option->style           = AF_OPTION_STYLE_AMERICAN;
  BOOST_CHECK(af_ResultIsUnknownFloat(af_BSOptionDelta(option)));
  af_OptionInfoDelete(option);
}

//
// Checks that the option gamma is calculated correctly.
//
BOOST_AUTO_TEST_CASE( BSOptionValuation_af_BSOptionGamma ) {
  afOptionInfo_t* option  = afTest_CreateOption(AF_OPTION_TYPE_CALL, 100., 100., 10. / 250., 0.05, 0.2, 0.);
  BOOST_CHECK(abs(af_BSOptionGamma(option) - 0.0995) < EPSILON);
  option->type            = AF_OPTION_TYPE_PUT;
  BOOST_CHECK(abs(af_BSOptionGamma(option) - 0.0995) < EPSILON);
  option->style           = AF_OPTION_STYLE_AMERICAN;
  BOOST_CHECK(af_ResultIsUnknownFloat(af_BSOptionDelta(option)));
  af_OptionInfoDelete(option);
}

//
// Checks that the option vega is calculated correctly.
//
BOOST_AUTO_TEST_CASE( BSOptionValuation_af_BSOptionVega ) {
  afOptionInfo_t* option  = afTest_CreateOption(AF_OPTION_TYPE_CALL, 100., 100., 10. / 250., 0.05, 0.2, 0.);
  BOOST_CHECK(abs(af_BSOptionVega(option) - 7.9593) < EPSILON);
  option->type            = AF_OPTION_TYPE_PUT;
  BOOST_CHECK(abs(af_BSOptionVega(option) - 7.9593) < EPSILON);
  option->style           = AF_OPTION_STYLE_AMERICAN;
  BOOST_CHECK(af_ResultIsUnknownFloat(af_BSOptionDelta(option)));
  af_OptionInfoDelete(option);
}

//
// Checks that the option rho is calculated correctly.
//
BOOST_AUTO_TEST_CASE( BSOptionValuation_af_BSOptionRho ) {
  afOptionInfo_t* option  = afTest_CreateOption(AF_OPTION_TYPE_CALL, 100., 100., 10. / 250., 0.05, 0.2, 0.);
  BOOST_CHECK(abs(af_BSOptionRho(option) - 2.0438) < EPSILON);
  option->type            = AF_OPTION_TYPE_PUT;
  BOOST_CHECK(abs(af_BSOptionRho(option) + 1.9482) < EPSILON);
  option->style           = AF_OPTION_STYLE_AMERICAN;
  BOOST_CHECK(af_ResultIsUnknownFloat(af_BSOptionDelta(option)));
  af_OptionInfoDelete(option);
}

//
// Checks that the option theta is calculated correctly.
//
BOOST_AUTO_TEST_CASE( BSOptionValuation_af_BSOptionTheta ) {
  afOptionInfo_t* option  = afTest_CreateOption(AF_OPTION_TYPE_CALL, 100., 100., 10. / 250., 0.05, 0.2, 0.);
  BOOST_CHECK(abs(af_BSOptionTheta(option) + 22.453) < EPSILON);
  option->type            = AF_OPTION_TYPE_PUT;
  BOOST_CHECK(abs(af_BSOptionTheta(option) + 17.463) < EPSILON);
  option->style           = AF_OPTION_STYLE_AMERICAN;
  BOOST_CHECK(af_ResultIsUnknownFloat(af_BSOptionDelta(option)));
  af_OptionInfoDelete(option);
}

#endif // AF_TEST_CUDA_BS_OPTION_VALUATION_H

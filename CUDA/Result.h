//
// Created by adam on 31/10/15.
//

#ifndef AF_RESULT_H
#define AF_RESULT_H

#include "math.h"
#include "time.h"
#include "stdbool.h"

#define AF_RESULT_SUCCESS 1
#define AF_RESULT_ERROR 0

#define AF_UNKNOWN_FLOAT NAN
#define AF_UNKNOWN_TIME (time_t) 0


bool af_ResultIsUnknownFloat(float x);

bool af_ResultIsUnknownTime(time_t x);

#endif //AF_RESULT_H

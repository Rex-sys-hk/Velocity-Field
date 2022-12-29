#pragma once


double NormalizeAngle(const double angle);

double InterpolateAngle(const double a0, const double t0, const double a1,
                        const double t1, const double t);

int RandomInt(const int size);

double RandomDouble(const double lb, const double ub);

double Curvature(const double dx, const double d2x, const double dy,
                 const double d2y);

double CurvatureDerivative(const double dx, const double d2x, const double d3x,
                           const double dy, const double d2y, const double d3y);

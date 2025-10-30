#include <iostream>
#include <vector>
#include <functional>

// подключение всех заголовочных фалов
#include "gradient_descent.h"
#include "newton.h"
#include "newton-raphson.h"
#include "fletcher-reeves.h"

using namespace std;

// основная функция
long double main_function(double x1, double x2) {
    return x1 * x1 + 6 * x2 * x2 + x1 * x2 + x1;
}

int main() {
    setlocale(LC_ALL, "ru");

    // начальная точка
    double x1_0 = 1.5, x2_0 = 1.1;
    cout << "Функция: f(x1, x2) = x1^2 + 6x2^2 + x1x2 + x1\n";
    cout << "Начальная точка: (" << x1_0 << ", " << x2_0 << ")\n";
    cout << "Начальное значение: " << main_function(x1_0, x2_0);

    // аналитическое решение
    double x1_analytical = -12.0 / 23.0;
    double x2_analytical = 1.0 / 23.0;
    double f_analytical = main_function(x1_analytical, x2_analytical);

    cout << "\n\nАналитическое решение:\n";
    cout << "x1* = " << x1_analytical << ", x2* = " << x2_analytical;
    cout << "\nf(x*) = " << f_analytical << "\n\n";

    // применение различных методов
    GradientDescent gd(main_function);
    vector<double> result_gd = gd.optimize(x1_0, x2_0);

    Newton newton(main_function);
    vector<double> result_newton = newton.optimize(x1_0, x2_0);

    NewtonRaphson nr(main_function);
    vector<double> result_nr = nr.optimize(x1_0, x2_0);

    FletcherReeves fr(main_function);
    vector<double> result_fr = fr.optimize(x1_0, x2_0);

    return 0;
}
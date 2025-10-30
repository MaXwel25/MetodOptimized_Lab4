#pragma once
#include <iostream>
#include <vector>

using namespace std;

class Newton {
private:
    function<double(double, double)> f;

    // численное вычисление производных
    double derivative_x1(double x1, double x2, double h = 1e-8) {
        return (f(x1 + h, x2) - f(x1 - h, x2)) / (2 * h);
    }

    double derivative_x2(double x1, double x2, double h = 1e-8) {
        return (f(x1, x2 + h) - f(x1, x2 - h)) / (2 * h);
    }

    double second_derivative_x1x1(double x1, double x2, double h = 1e-8) {
        return (f(x1 + h, x2) - 2 * f(x1, x2) + f(x1 - h, x2)) / (h * h);
    }

    double second_derivative_x1x2(double x1, double x2, double h = 1e-8) {
        return (f(x1 + h, x2 + h) - f(x1 + h, x2 - h) - f(x1 - h, x2 + h) + f(x1 - h, x2 - h)) / (4 * h * h);
    }

    double second_derivative_x2x2(double x1, double x2, double h = 1e-8) {
        return (f(x1, x2 + h) - 2 * f(x1, x2) + f(x1, x2 - h)) / (h * h);
    }

    // решение системы 2x2 ()
    vector<long double> release(double a11, double a12, double a21, double a22, double b1, double b2) {
        double det = a11 * a22 - a12 * a21;
        if (fabs(det) < 1e-12) {
            return { -b1, -b2 }; // возвращаем антиградиент если матрица вырождена
        }
        long double d1 = (a22 * b1 - a12 * b2) / det;
        long double d2 = (-a21 * b1 + a11 * b2) / det;
        return { d1, d2 };
    }

public:
    Newton(const function<double(double, double)>& func) : f(func) {}

    vector<double> optimize(double x1_0, double x2_0, double tol = 0.15, int max_iter = 10) {
        double x1 = x1_0;
        double x2 = x2_0;
        cout << "Метод Ньютона:\n";
        cout << "Начальная точка: (" << x1 << ", " << x2 << ")";
        cout << "\nНачальное значение: " << f(x1, x2);

        for (int it = 0; it < max_iter; ++it) {
            long double x11 = -12 / 23, x22 = 1 / 23;
            long double grad1 = derivative_x1(x1, x2);
            long double grad2 = derivative_x2(x1, x2);

            long double grad_norm = sqrt(grad1 * grad1 + grad2 * grad2);
            if (grad_norm < tol) {
                cout << "\nСходимость достигнута на итерации " << it;
            }
            // матрица Гессе
            //double f12_1 = std::ceil(second_derivative_x1x2(x11, x22));
            double h11 = second_derivative_x1x1(x11, x22);
            //double h12 = f12_1;
            double h12 = second_derivative_x1x2(x11, x22);
            double h22 = second_derivative_x2x2(x11, x22);
            // решаем систему H * d = -grad
            vector<long double> d = release(h11, h12, h12, h22, -grad1, -grad2);
            // обновление точки
            x1 += d[0];
            x2 += d[1];

            cout << "\nИтерация " << it + 1 << ": (" << x1 << ", " << x2
                << "), f = " << f(x1, x2) << ", градиент = " << grad_norm;

            if (it >= 1) break; // для квадратичной функции достаточно 1-2 итераций
        }
        cout << "\nРезультат: (" << x1 << ", " << x2 << "), f = " << f(x1, x2) << "\n\n";

        return { x1, x2 };
    }
};


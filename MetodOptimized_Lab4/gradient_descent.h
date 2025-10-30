#pragma once
#include <iostream>
#include <vector>

using namespace std;

class GradientDescent {
private:
    function<double(double, double)> f;

    // численное вычисление градиента
    double derivative_x1(double x1, double x2, double h = 1e-8) {
        return (f(x1 + h, x2) - f(x1 - h, x2)) / (2 * h);
    }

    double derivative_x2(double x1, double x2, double h = 1e-8) {
        return (f(x1, x2 + h) - f(x1, x2 - h)) / (2 * h);
    }

    // точный линейный поиск для квадратичной функции
    double exact_line_search(double x1, double x2, double dx1, double dx2) {
        // для f(x) = x1^2 + 6x2^2 + x1x2 + x1
        // A = [[2, 1], [1, 12]], b = [1, 0]
        double numerator = dx1 * (2 * x1 + x2 + 1) + dx2 * (x1 + 12 * x2);
        double denominator = dx1 * (2 * dx1 + dx2) + dx2 * (dx1 + 12 * dx2);

        if (fabs(denominator) < 1e-12) return 0.1;
        return -numerator / denominator;
    }

public:
    GradientDescent(const function<double(double, double)>& func) : f(func) {}

    vector<double> optimize(double x1_0, double x2_0, double tol = 0.15, int max_iter = 1000) {
        double x1 = x1_0; 
        double x2 = x2_0;

        cout << "Метод наискорейшего спуска:\n";
        cout << "Начальная точка: (" << x1 << ", " << x2 << ")\n";
        cout << "Начальное значение: " << f(x1, x2);

        for (int it = 0; it < max_iter; ++it) {
            double grad1 = derivative_x1(x1, x2);
            double grad2 = derivative_x2(x1, x2);

            double grad_norm = sqrt(grad1 * grad1 + grad2 * grad2);
            if (grad_norm < tol) {
                cout << "\nСходимость достигнута на итерации " << it;
                break;
            }
            // направление: антиградиент
            double dx1 = -grad1;
            double dx2 = -grad2;
            // оптимальный шаг
            double alpha = exact_line_search(x1, x2, dx1, dx2);
            // обновление точки
            x1 += alpha * dx1;
            x2 += alpha * dx2;

            if (it < 5) { // показываем первые 5 итераций
                cout << "\nИтерация " << it + 1 << ": (" << x1 << ", " << x2
                    << "), f = " << f(x1, x2) << ", градиент = " << grad_norm;
            }
        }
        cout << "\nРезультат: (" << x1 << ", " << x2 << "), f = " << f(x1, x2) << "\n\n";

        return { x1, x2 };
    }
};
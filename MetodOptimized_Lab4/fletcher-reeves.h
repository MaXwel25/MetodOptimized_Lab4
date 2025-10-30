#pragma once
#include <iostream>
#include <vector>

using namespace std;

class FletcherReeves {
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
        double numerator = dx1 * (2 * x1 + x2 + 1) + dx2 * (x1 + 12 * x2);
        double denominator = dx1 * (2 * dx1 + dx2) + dx2 * (dx1 + 12 * dx2);

        if (fabs(denominator) < 1e-12) return 0.1;
        return -numerator / denominator;
    }

public:
    FletcherReeves(const function<double(double, double)>& func) : f(func) {}

    vector<double> optimize(double x1_0, double x2_0, double tol = 0.15, int max_iter = 1000) {
        double x1 = x1_0;
        double x2 = x2_0;

        cout << "Метод Флетчера-Ривса:" << "\n";
        cout << "Начальная точка: (" << x1 << ", " << x2 << ")\n";
        cout << "Начальное значение: " << f(x1, x2);

        // начальный градиент и направление
        double grad1 = derivative_x1(x1, x2);
        double grad2 = derivative_x2(x1, x2);
        double dx1 = -grad1;
        double dx2 = -grad2;

        for (int iter = 0; iter < max_iter; ++iter) {
            double grad_norm = sqrt(grad1 * grad1 + grad2 * grad2);
            if (grad_norm < tol) {
                cout << "\nСходимость достигнута на итерации " << iter;
                break;
            }
            // сохраняем старый градиент
            double grad1_old = grad1;
            double grad2_old = grad2;
            // оптимальный шаг
            double alpha = exact_line_search(x1, x2, dx1, dx2);
            // обновление точки
            x1 += alpha * dx1;
            x2 += alpha * dx2;
            // новый градиент
            grad1 = derivative_x1(x1, x2);
            grad2 = derivative_x2(x1, x2);
            // коэффициент Флетчера-Ривса
            double beta = (grad1 * grad1 + grad2 * grad2) / (grad1_old * grad1_old + grad2_old * grad2_old);
            // новое сопряженное направление
            dx1 = -grad1 + beta * dx1;
            dx2 = -grad2 + beta * dx2;
            cout << "\nИтерация " << iter + 1 << ": (" << x1 << ", " << x2
                << "), f = " << f(x1, x2) << ", градиент = " << grad_norm;

            //if (iter >= 2) break; // для квадратичной функции с n=2 сходится за n итераций
        }
        cout << "\nРезультат: (" << x1 << ", " << x2 << "), f = " << f(x1, x2) << "\n";

        return { x1, x2 };
    }
};

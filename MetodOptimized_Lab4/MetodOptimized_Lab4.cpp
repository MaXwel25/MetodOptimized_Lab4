#include <iostream>
#include <cmath>
#include <vector>

using namespace std;

// функция f(x) = x1^2 + 6*x2^2 + x1*x2 + x1
double main_func(const vector<double>& x) {
    return x[0] * x[0] + 6 * x[1] * x[1] + x[0] * x[1] + x[0];
}

// градиент функции
vector<double> grad_func(const vector<double>& x) {
    vector<double> grad(2);
    grad[0] = 2 * x[0] + x[1] + 1;  // df/dx1
    grad[1] = 12 * x[1] + x[0];     // df/dx2
    return grad;
}

// матрица Гессе (вместо ручного подсчёта)
vector<vector<double>> hessian_func() {
    vector<vector<double>> H(2, vector<double>(2));
    H[0][0] = 2;  // d^2f/dx1^2
    H[0][1] = 1;  // d^2f/dx1dx2
    H[1][0] = 1;  // d^2f/dx2dx1
    H[1][1] = 12; // d^2f/dx2^2
    return H;
}

// скалярное произведение векторов
double dot_product(const vector<double>& a, const vector<double>& b) {
    return a[0] * b[0] + a[1] * b[1];
}

// норма вектора
double vector_norm(const vector<double>& v) {
    return sqrt(v[0] * v[0] + v[1] * v[1]);
}

// умножение матрицы на вектор
vector<double> matrix_vector_mult(const vector<vector<double>>& M, const vector<double>& v) {
    vector<double> result(2);
    result[0] = M[0][0] * v[0] + M[0][1] * v[1];
    result[1] = M[1][0] * v[0] + M[1][1] * v[1];
    return result;
}

// метод наискорейшего спуска
vector<double> steepest_descent(const vector<double>& x0, double eps1, double eps2, int M) {
    vector<double> x = x0;
    vector<vector<double>> H = hessian_func();

    cout << "Метод наискорейшего спуска:" << endl;
    cout << "--------------------------------------------------------" << endl;
    for (int k = 0; k < M; k++) {
        vector<double> grad = grad_func(x);
        double grad_norm = vector_norm(grad);
        double fx = main_func(x);

        cout << "Итерация: " << k << endl;
        cout << "x0: " << x[0] << endl;
        cout << "x1: " << x[1] << endl;
        cout << "Значение функции: " << fx << endl;
        cout << "Норма градиета: " << grad_norm << endl;

        // шаг 4: проверка критерия окончания по градиенту
        if (grad_norm < eps1) {
            cout << "Критерий окончания по градиенту выполнен!" << endl;
            break;
        }

        // шаг 6: Вычисление оптимального шага
        vector<double> H_grad = matrix_vector_mult(H, grad);
        double numerator = dot_product(grad, grad);
        double denominator = dot_product(grad, H_grad);
        double t = numerator / denominator;

        // сохраняем предыдущую точку для проверки сходимости
        vector<double> x_prev = x;

        // шаг 7: вычисление новой точки
        x[0] = x[0] - t * grad[0];
        x[1] = x[1] - t * grad[1];

        // шаг 8: проверка условий сходимости
        double delta_x_norm = vector_norm({ x[0] - x_prev[0], x[1] - x_prev[1] });
        double delta_f = abs(main_func(x) - main_func(x_prev));

        if (delta_x_norm < eps2 && delta_f < eps2) {
            // вывод финальной итерации
            cout << "--------------------------------------------------------" << endl;
            cout << "Итерация: " << k + 1 << endl; // итерация
            cout << "x0: " << x[0]<<endl;
            cout << "x1: " << x[1] << endl;
            cout << "Значение функции: "<< main_func(x) << endl;
            cout << "Норма градиента: " << vector_norm(grad_func(x)) << endl; // градиент
            cout << "Критерий окончания по изменению точки и функции выполнен!" << endl;
            break;
        }

        // на случай переполнения
        if (k == M - 1) {
            cout << "Достигнуто максимальное число итераций!" << endl;
        }
        cout << "--------------------------------------------------------" << endl;
    }
    cout << "--------------------------------------------------------" << endl;
    return x;
}

int main() {
    setlocale(LC_ALL, "ru");
    vector<double> x0 = { 1.5, 1.1 };

    // параметры алгоритма
    double eps1 = 0.1;  // точность по градиенту
    double eps2 = 0.15; // точность по изменению точки и функции
    int M = 10;  // максимальное число итераций

    cout << "Минимизация функции f(x) = x1^2 + 6*x2^2 + x1*x2 + x1" << endl;
    cout << "Начальная точка: x0 = (" << x0[0] << ", " << x0[1] << ")" << endl;
    cout << "Начальное значение функции: f(x0) = " << main_func(x0) << endl;
    cout << "Параметры: eps1 = " << eps1 << ", eps2 = " << eps2 << ", M = " << M << endl;
    cout << endl;

    // запуск метода наискорейшего спуска
    vector<double> x_opt = steepest_descent(x0, eps1, eps2, M);

    cout << endl;
    cout << "Результат:" << endl;
    cout << "Оптимальная точка: x* = (" << x_opt[0] << ", " << x_opt[1] << ")" << endl;
    cout << "Значение функции в оптимальной точке: f(x*) = " << main_func(x_opt) << endl;
    cout << "Градиент в оптимальной точке: *f(x*) = ("
        << grad_func(x_opt)[0] << ", " << grad_func(x_opt)[1] << ")" << endl;
    cout << "Норма градиента: ||*f(x*)|| = " << vector_norm(grad_func(x_opt)) << endl;

    return 0;
}
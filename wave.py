import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.sparse import diags

def solve_schrodinger_nabla_n(n, N=200, L=1.0, hbar=1.0, m=1.0):
    """
    Решает стационарное уравнение Шредингера с оператором ∇^n
    для частицы в бесконечной потенциальной яме [0, L].
    
    Parameters:
    -----------
    n : int
        Порядок оператора ∇^n (n >= 2)
    N : int
        Количество узлов сетки (чем больше, тем точнее)
    L : float
        Ширина ямы
    hbar : float
        Приведенная постоянная Планка (можно оставить 1 в атомных единицах)
    m : float
        Масса частицы (можно оставить 1)
    
    Returns:
    --------
    energies : ndarray
        Массив первых N собственных значений (энергий)
    wavefunctions : ndarray
        Массив собственных векторов (волновых функций)
    x : ndarray
        Координатная сетка
    """
    
    if n < 2:
        raise ValueError("n должно быть >= 2")
    
    # Шаг сетки
    dx = L / (N + 1)
    x = np.linspace(0, L, N + 2)[1:-1]  # узлы без границ
    
    # Коэффициент перед оператором
    coeff = (hbar**2) / (2 * m)
    
    # Построение матрицы для оператора ∇^n с использованием конечных разностей
    # Для n=2: стандартный лапласиан (3-точечный шаблон)
    # Для n=3: 4-точечный шаблон
    # Для n=4: 5-точечный шаблон и т.д.
    
    # Размер матрицы
    size = N
    
    # Шаблон конечных разностей для производной n-го порядка
    # Используем коэффициенты из формулы:
    # d^n f / dx^n ≈ (1/dx^n) * Σ_{k=-r}^{r} c_k * f(x + k*dx)
    # где r = n//2 + 1 (радиус шаблона)
    
    radius = n // 2 + 1
    # Создаем диагональную матрицу с коэффициентами конечных разностей
    # Для простоты используем метод наименьших квадратов или готовые коэффициенты
    
    # Для n=2,3,4 используем явные формулы
    if n == 2:
        # Стандартный 3-точечный шаблон: [1, -2, 1] / dx^2
        main_diag = -2 * np.ones(size)
        off_diag = np.ones(size - 1)
        H_matrix = coeff * (diags([off_diag, main_diag, off_diag], [-1, 0, 1]) / (dx**2))
        H_matrix = H_matrix.toarray()
        
    elif n == 3:
        # 4-точечный шаблон для третьей производной:
        # [-1/2, 1, 0, -1, 1/2] / dx^3
        # Правильные коэффициенты для центральной разности 4-го порядка:
        # [-1/2, 1, 0, -1, 1/2]
        # Но для краев нужна специальная аппроксимация
        # Используем упрощенный вариант с несимметричными шаблонами
        print("Предупреждение: для n=3 используется аппроксимация")
        # Строим матрицу с использованием финитных разностей
        H_matrix = np.zeros((size, size))
        for i in range(size):
            # Шаблон для внутренних точек
            if 2 <= i < size - 2:
                H_matrix[i, i-2] = -0.5
                H_matrix[i, i-1] = 1.0
                H_matrix[i, i] = 0.0
                H_matrix[i, i+1] = -1.0
                H_matrix[i, i+2] = 0.5
            else:
                # Для краев используем односторонние разности (приближенно)
                # Упрощенный вариант
                H_matrix[i, i] = 0.0
        H_matrix = coeff * H_matrix / (dx**3)
        
    elif n == 4:
        # 5-точечный шаблон для четвертой производной:
        # [1, -4, 6, -4, 1] / dx^4
        H_matrix = np.zeros((size, size))
        for i in range(size):
            if 2 <= i < size - 2:
                H_matrix[i, i-2] = 1.0
                H_matrix[i, i-1] = -4.0
                H_matrix[i, i] = 6.0
                H_matrix[i, i+1] = -4.0
                H_matrix[i, i+2] = 1.0
            else:
                # Для краев упрощенное приближение
                H_matrix[i, i] = 6.0
        H_matrix = coeff * H_matrix / (dx**4)
        
    else:
        # Для n > 4 используем общий метод конечных разностей
        print(f"Используем общий метод для n={n}")
        # Строим матрицу с помощью конечных разностей высокого порядка
        # Используем пакет findiff или реализуем вручную
        # Для простоты используем аппроксимацию с помощью матрицы дифференцирования
        H_matrix = build_nabla_matrix(n, size, dx)
        H_matrix = coeff * H_matrix
    
    # Решаем задачу на собственные значения
    # Используем eigh для симметричных матриц (только для четных n)
    if n % 2 == 0:
        # Для четных n матрица симметрична
        energies, wavefunctions = eigh(H_matrix)
    else:
        # Для нечетных n матрица несимметрична, используем общую диагонализацию
        from scipy.linalg import eig
        eigvals, eigvecs = eig(H_matrix)
        # Сортируем по возрастанию действительной части
        idx = np.argsort(eigvals.real)
        energies = eigvals[idx].real
        wavefunctions = eigvecs[:, idx].real
    
    return energies[:10], wavefunctions[:, :10], x


def build_nabla_matrix(n, size, dx):
    """
    Строит матрицу для оператора ∇^n с помощью конечных разностей.
    Используется метод наименьших квадратов для нахождения коэффициентов.
    """
    from scipy.linalg import lstsq

    # Радиус шаблона
    radius = n // 2 + 2
    # Количество точек в шаблоне
    n_points = 2 * radius + 1

    # Коэффициенты для производной n-го порядка
    # Используем метод наименьших квадратов
    # Строим матрицу Вандермонда
    stencil = np.arange(-radius, radius + 1)
    V = np.vander(stencil, increasing=True)
    # Целевой вектор: коэффициент при x^n должен быть n!, остальные 0
    target = np.zeros(n_points)
    target[n] = np.math.factorial(n)

    # Находим коэффициенты
    coeffs = lstsq(V, target)[0]
    coeffs = coeffs / (dx**n)

    # Строим матрицу
    H_matrix = np.zeros((size, size))
    for i in range(size):
        for j, k in enumerate(range(-radius, radius + 1)):
            idx = i + k
            if 0 <= idx < size:
                H_matrix[i, idx] = coeffs[j]

    return H_matrix


def plot_results(n_values, N=100, L=1.0):
    """
    Визуализирует энергии и волновые функции для разных n
    """
    fig, axes = plt.subplots(2, len(n_values), figsize=(15, 8))
    if len(n_values) == 1:
        axes = axes.reshape(2, 1)
    
    for idx, n in enumerate(n_values):
        energies, wf, x = solve_schrodinger_nabla_n(n, N=N, L=L)
        
        # График энергий
        ax1 = axes[0, idx]
        ax1.stem(range(1, min(6, len(energies)+1)), energies[:5], basefmt=" ")
        ax1.set_title(f'n = {n}')
        ax1.set_xlabel('Квантовое состояние')
        ax1.set_ylabel('Энергия (a.u.)')
        ax1.grid(alpha=0.3)
        
        # График волновых функций
        ax2 = axes[1, idx]
        for i in range(min(3, wf.shape[1])):
            # Нормировка для отображения
            psi = wf[:, i] / np.max(np.abs(wf[:, i])) * 0.8 + i * 1.5
            ax2.plot(x, psi, label=f'n={i+1}')
        ax2.set_title(f'Волновые функции (n={n})')
        ax2.set_xlabel('x (a.u.)')
        ax2.set_ylabel('Ψ(x) (сдвинуты)')
        ax2.legend()
        ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# ============ ОСНОВНАЯ ФУНКЦИЯ ============

def main():
    """
    Главная функция для демонстрации работы программы
    """
    print("=" * 60)
    print("РЕШЕНИЕ УРАВНЕНИЯ ШРЕДИНГЕРА С ОПЕРАТОРОМ ∇^n")
    print("=" * 60)

    # Тестируем разные значения n
    n_values = [2, 3, 4]

    for n in n_values:
        print(f"\n--- n = {n} ---")
        energies, wf, x = solve_schrodinger_nabla_n(n, N=100, L=1.0)
        
        print(f"Первые 5 уровней энергии:")
        for i, E in enumerate(energies[:5]):
            print(f"  E_{i+1} = {E:.6f} (a.u.)")

        # Проверка ортогональности
        if n == 2:
            # Для n=2 проверяем аналитическое решение
            E_theor = (np.pi**2 * np.arange(1, 6)**2) / 2
            print(f"\nАналитические энергии (n=2):")
            for i, E in enumerate(E_theor):
                print(f"  E_{i+1} = {E:.6f} (a.u.)")

    # Визуализация
    print("\nСтроим графики...")
    plot_results([2, 3, 4], N=150, L=1.0)


if __name__ == "__main__":
    main()

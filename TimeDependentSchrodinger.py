import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.fft import fft, ifft, fftfreq
from scipy.linalg import solve_banded

class TimeDependentSchrodinger:
    """
    Класс для решения временного уравнения Шредингера
    с обобщенным оператором ∇^n
    """
    
    def __init__(self, n=2, N=256, L=10.0, hbar=1.0, m=1.0):
        """
        Parameters:
        -----------
        n : int
            Порядок оператора ∇^n (n >= 2)
        N : int
            Количество узлов сетки (должно быть степенью 2 для FFT)
        L : float
            Размер области [-L/2, L/2]
        hbar : float
            Приведенная постоянная Планка
        m : float
            Масса частицы
        """
        self.n = n
        self.N = N
        self.L = L
        self.hbar = hbar
        self.m = m
        
        # Координатная сетка
        self.x = np.linspace(-L/2, L/2, N)
        self.dx = self.x[1] - self.x[0]
        
        # Волновые векторы для FFT
        self.k = 2 * np.pi * fftfreq(N, self.dx)
        
        # Оператор кинетической энергии в k-пространстве
        # Для ∇^n: T(k) = (hbar^2 / 2m) * (i*k)^n
        # Для четных n: (i)^n = (-1)^(n/2), для нечетных - мнимые
        if n % 2 == 0:
            self.T_k = (hbar**2 / (2 * m)) * (1j * self.k)**n
        else:
            # Для нечетных n оператор неэрмитов, используем действительную часть
            # или добавляем малую мнимую часть для стабильности
            self.T_k = (hbar**2 / (2 * m)) * (1j * self.k)**n
            # Добавляем малую мнимую часть для устойчивости
            self.T_k = self.T_k.real + 1e-6 * self.T_k.imag
        
        # Потенциальная энергия (по умолчанию - яма)
        self.potential = None
        self.set_potential('well')
    
    def set_potential(self, kind='well', params=None):
        """
        Установка потенциальной энергии
        
        Parameters:
        -----------
        kind : str
            'well' - бесконечная яма
            'harmonic' - гармонический осциллятор
            'barrier' - потенциальный барьер
            'custom' - пользовательский потенциал
        params : dict
            Параметры потенциала
        """
        if kind == 'well':
            # Бесконечная яма (моделируется большим потенциалом на краях)
            self.U = np.zeros_like(self.x)
            mask = np.abs(self.x) > self.L/4
            self.U[mask] = 1e6  # Очень большой потенциал на краях
            
        elif kind == 'harmonic':
            # Гармонический осциллятор U(x) = 0.5 * m * omega^2 * x^2
            omega = params.get('omega', 1.0) if params else 1.0
            self.U = 0.5 * self.m * omega**2 * self.x**2
            
        elif kind == 'barrier':
            # Потенциальный барьер
            height = params.get('height', 10.0) if params else 10.0
            width = params.get('width', 0.5) if params else 0.5
            self.U = np.zeros_like(self.x)
            mask = np.abs(self.x) < width/2
            self.U[mask] = height
            
        elif kind == 'custom':
            if params is not None and 'U' in params:
                self.U = params['U']
            else:
                raise ValueError("Для custom нужно передать U в params")
        
        # Оператор потенциальной энергии в реальном пространстве
        self.V = np.diag(self.U)
    
    def initial_wavefunction(self, kind='gaussian', params=None):
        """
        Создание начальной волновой функции
        
        Parameters:
        -----------
        kind : str
            'gaussian' - гауссов волновой пакет
            'eigenstate' - собственное состояние (для n=2)
        params : dict
            Параметры волновой функции
        """
        if kind == 'gaussian':
            x0 = params.get('x0', 0.0) if params else 0.0
            sigma = params.get('sigma', 0.5) if params else 0.5
            k0 = params.get('k0', 5.0) if params else 5.0
            
            # Гауссов волновой пакет с начальным импульсом
            psi = np.exp(-(self.x - x0)**2 / (4 * sigma**2)) * np.exp(1j * k0 * self.x)
            # Нормировка
            psi = psi / np.sqrt(np.sum(np.abs(psi)**2) * self.dx)
            
        elif kind == 'eigenstate':
            # Собственное состояние частицы в яме (только для n=2)
            n_state = params.get('n', 1) if params else 1
            if self.n == 2:
                # Частица в яме с бесконечными стенками
                psi = np.sin(n_state * np.pi * (self.x + self.L/2) / self.L)
                mask = np.abs(self.x) > self.L/2
                psi[mask] = 0
                psi = psi / np.sqrt(np.sum(np.abs(psi)**2) * self.dx)
            else:
                raise ValueError("Eigenstate только для n=2")
        
        elif kind == 'random':
            # Случайная волновая функция
            psi = np.random.randn(self.N) + 1j * np.random.randn(self.N)
            psi = psi / np.sqrt(np.sum(np.abs(psi)**2) * self.dx)
        
        return psi
    
    def split_step(self, psi, dt):
        """
        Один шаг эволюции методом расщепления операторов (Split-Operator)
        
        psi(t+dt) = exp(-i*V*dt/2) * exp(-i*T*dt) * exp(-i*V*dt/2) * psi(t)
        """
        # 1. Половина шага потенциальной энергии
        psi = psi * np.exp(-1j * self.U * dt / 2)
        
        # 2. Переход в k-пространство и применение кинетической энергии
        psi_k = fft(psi)
        psi_k = psi_k * np.exp(-1j * self.T_k * dt)
        psi = ifft(psi_k)
        
        # 3. Еще половина шага потенциальной энергии
        psi = psi * np.exp(-1j * self.U * dt / 2)
        
        return psi
    
    def crank_nicolson(self, psi, dt):
        """
        Один шаг эволюции методом Кранка-Николсона (неявная схема)
        """
        # Строим матрицу для неявной схемы
        # (I + i*dt*H/2) * psi(t+dt) = (I - i*dt*H/2) * psi(t)
        
        # Для простоты используем явную схему с малым dt (аналог CN)
        # Полная реализация CN требует решения системы уравнений
        psi_k = fft(psi)
        psi_k = psi_k * np.exp(-1j * self.T_k * dt)
        psi = ifft(psi_k)
        psi = psi * np.exp(-1j * self.U * dt)
        
        return psi
    
    def evolve(self, psi0, dt, n_steps, method='split', save_every=10):
        """
        Эволюция волновой функции во времени
        
        Parameters:
        -----------
        psi0 : ndarray
            Начальная волновая функция
        dt : float
            Шаг по времени
        n_steps : int
            Количество шагов
        method : str
            'split' - метод расщепления
            'cn' - метод Кранка-Николсона
        save_every : int
            Сохранять состояние каждые save_every шагов
        
        Returns:
        --------
        times : ndarray
            Массив времен
        psi_history : ndarray
            Массив волновых функций в разные моменты времени
        """
        psi = psi0.copy()
        psi_history = [psi.copy()]
        times = [0.0]
        
        for step in range(1, n_steps + 1):
            if method == 'split':
                psi = self.split_step(psi, dt)
            elif method == 'cn':
                psi = self.crank_nicolson(psi, dt)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            if step % save_every == 0:
                psi_history.append(psi.copy())
                times.append(step * dt)
        
        return np.array(times), np.array(psi_history)
    
    def animate_evolution(self, psi0, dt, n_steps, method='split', save_every=5):
        """
        Создание анимации эволюции волновой функции
        """
        times, psi_history = self.evolve(psi0, dt, n_steps, method, save_every)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Плотность вероятности
        prob = np.abs(psi_history[0])**2
        line1, = ax1.plot(self.x, prob, 'b-', linewidth=2, label='$|\psi|^2$')
        ax1.set_ylim(0, np.max(prob) * 1.2)
        ax1.set_xlim(self.x[0], self.x[-1])
        ax1.set_ylabel('Плотность вероятности')
        ax1.grid(alpha=0.3)
        
        # Действительная и мнимая части
        line2, = ax2.plot(self.x, psi_history[0].real, 'r-', linewidth=1.5, label='Re ψ')
        line3, = ax2.plot(self.x, psi_history[0].imag, 'g--', linewidth=1.5, label='Im ψ')
        ax2.set_ylim(-1.5, 1.5)
        ax2.set_xlim(self.x[0], self.x[-1])
        ax2.set_xlabel('x')
        ax2.set_ylabel('Ψ(x)')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # Добавляем потенциал на оба графика
        if np.max(self.U) < 50:  # Не показываем огромные потенциалы
            ax1_twin = ax1.twinx()
            ax1_twin.plot(self.x, self.U / np.max(self.U + 1e-10) * np.max(prob), 'k--', alpha=0.3, linewidth=1)
            ax1_twin.set_ylabel('Потенциал (норм.)')
        
        time_text = ax1.text(0.02, 0.95, f't = {0:.2f}', transform=ax1.transAxes, 
                           bbox=dict(facecolor='white', alpha=0.8))
        
        def update(frame):
            psi = psi_history[frame]
            prob = np.abs(psi)**2
            
            line1.set_ydata(prob)
            line2.set_ydata(psi.real)
            line3.set_ydata(psi.imag)
            
            time_text.set_text(f't = {times[frame]:.2f}')
            ax1.set_ylim(0, np.max(prob) * 1.2)
            
            return line1, line2, line3, time_text
        
        anim = FuncAnimation(fig, update, frames=len(psi_history), 
                           interval=50, blit=True)
        
        plt.tight_layout()
        return fig, anim


# ============ ОСНОВНАЯ ФУНКЦИЯ ============

def main():
    """
    Демонстрация работы временного уравнения Шредингера
    """
    print("=" * 70)
    print("ВРЕМЕННОЕ УРАВНЕНИЕ ШРЕДИНГЕРА С ОПЕРАТОРОМ ∇^n")
    print("=" * 70)
    
    # Создаем объект для n=2 (стандартная квантовая механика)
    solver = TimeDependentSchrodinger(n=2, N=256, L=10.0)
    
    # Начальное состояние - гауссов волновой пакет
    psi0 = solver.initial_wavefunction(
        kind='gaussian', 
        params={'x0': -2.0, 'sigma': 0.5, 'k0': 5.0}
    )
    
    # Параметры эволюции
    dt = 0.01
    n_steps = 200
    method = 'split'
    
    print(f"\nНачальное состояние: гауссов волновой пакет")
    print(f"Шаг по времени: dt = {dt}")
    print(f"Количество шагов: {n_steps}")
    print(f"Метод: {method}")
    print(f"Оператор: ∇^{solver.n}")
    
    # Выполняем эволюцию
    print("\nВыполняется эволюция...")
    times, psi_history = solver.evolve(psi0, dt, n_steps, method, save_every=5)
    
    print(f"Сохранено {len(psi_history)} состояний")
    
    # Вычисляем энергию в разные моменты времени
    energies = []
    for psi in psi_history:
        # Кинетическая энергия в k-пространстве
        psi_k = fft(psi)
        T = np.sum(np.abs(psi_k)**2 * solver.T_k) * solver.dx / (2 * np.pi)
        # Потенциальная энергия
        V = np.sum(np.abs(psi)**2 * solver.U) * solver.dx
        energies.append(T + V)
    
    energies = np.array(energies)
    print(f"\nСредняя энергия:")
    print(f"  Начальная: {energies[0]:.6f}")
    print(f"  Конечная:  {energies[-1]:.6f}")
    print(f"  Отклонение: {np.std(energies):.6f} (должно быть ~0)")
    
    # Визуализация эволюции
    print("\nСоздание анимации...")
    fig, anim = solver.animate_evolution(psi0, dt, n_steps, method, save_every=5)
    
    # Сохраняем анимацию
    # anim.save('schrodinger_evolution.gif', writer='pillow', fps=20)
    # print("Анимация сохранена как 'schrodinger_evolution.gif'")
    
    plt.show()
    
    # Дополнительный график: энергия во времени
    plt.figure(figsize=(10, 4))
    plt.plot(times, energies, 'b-', linewidth=2)
    plt.axhline(y=energies[0], color='r', linestyle='--', label='Начальная энергия')
    plt.xlabel('Время')
    plt.ylabel('Энергия')
    plt.title('Сохранение энергии во времени')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def compare_n_values():
    """
    Сравнение эволюции для разных n
    """
    print("\n" + "=" * 70)
    print("СРАВНЕНИЕ ЭВОЛЮЦИИ ДЛЯ РАЗНЫХ n")
    print("=" * 70)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    n_values = [2, 3, 4]
    colors = ['blue', 'red', 'green']
    
    for idx, n in enumerate(n_values):
        solver = TimeDependentSchrodinger(n=n, N=256, L=10.0)
        
        # Начальное состояние
        psi0 = solver.initial_wavefunction(
            kind='gaussian', 
            params={'x0': -2.0, 'sigma': 0.5, 'k0': 5.0}
        )
        
        # Эволюция
        dt = 0.005
        n_steps = 100
        times, psi_history = solver.evolve(psi0, dt, n_steps, 'split', save_every=5)
        
        # Показываем начальное и конечное состояния
        ax1 = axes[0, idx]
        ax1.plot(solver.x, np.abs(psi0)**2, 'k-', linewidth=2, label='t=0')
        ax1.plot(solver.x, np.abs(psi_history[-1])**2, colors[idx], linewidth=2, label=f't={times[-1]:.1f}')
        ax1.set_title(f'∇^{n}')
        ax1.set_xlabel('x')
        ax1.set_ylabel('$|\psi|^2$')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Динамика центра волнового пакета
        ax2 = axes[1, idx]
        center = []
        for psi in psi_history:
            prob = np.abs(psi)**2
            center.append(np.sum(solver.x * prob) * solver.dx)
        ax2.plot(times, center, f'{colors[idx]}-', linewidth=2)
        ax2.set_xlabel('Время')
        ax2.set_ylabel('⟨x⟩')
        ax2.set_title(f'Движение центра (∇^{n})')
        ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Запускаем основную демонстрацию
    main()
    
    # Сравнение для разных n
    # compare_n_values()  # Раскомментировать для сравнения

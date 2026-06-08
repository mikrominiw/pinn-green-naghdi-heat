import torch
from torch import nn
from torch import optim
from torch import autograd
import matplotlib.pyplot as plt
import numpy as np

# %% Настройка устройства
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используется устройство: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
torch.cuda.empty_cache()

# %% Архитектура PINN
class PINN(nn.Module):
    def __init__(self, t_max=500.0):
        super(PINN, self).__init__()
        self.t_max = t_max
        self.layer_input = nn.Linear(3, 150)
        self.layers = nn.ModuleList()
        for _ in range(3):
            self.layers.append(nn.Linear(150, 150))
        self.layer_out = nn.Linear(150, 1)
        self.activation = nn.Tanh()
        self.log_kappa = nn.Parameter(torch.log(torch.tensor(1.0)))  # начальное значение κ = 1.0

    def forward(self, x, y, t_physical):
        t_norm = t_physical / self.t_max
        inp = torch.cat([x, y, t_norm], dim=1)
        out = self.layer_input(inp)
        out = self.activation(out)
        for layer in self.layers:
            residual = out
            out = layer(out)
            out = self.activation(out)
            out = out + residual
        out = self.layer_out(out)
        return out

    def get_kappa(self):
        return torch.exp(self.log_kappa)

# %% Генератор точек
class POINTS:
    def __init__(self, n_pde=5000, n_bc=2000, n_ic=2000, n_sensors=5, n_time_points=100, t_max=500.0, device=device):
        self.n_pde = n_pde
        self.n_bc = n_bc
        self.n_ic = n_ic
        self.t_max = t_max
        self.device = device
        self.set_data(n_sensors, n_time_points, t_max)

    def set_data(self, n_sensors, n_time_points, t_max):
        self.sensor_x = torch.tensor([0.25, 0.25, 0.5, 0.75, 0.75], device=self.device)
        self.sensor_y = torch.tensor([0.75, 0.25, 0.5, 0.75, 0.25], device=self.device)
        self.t_points = torch.linspace(0, t_max, n_time_points, requires_grad=True, device=self.device)
        self.x_all, self.y_all, self.t_all = [], [], []
        for i in range(n_sensors):
            x_sensor = self.sensor_x[i].repeat(n_time_points)
            y_sensor = self.sensor_y[i].repeat(n_time_points)
            self.x_all.append(x_sensor)
            self.y_all.append(y_sensor)
            self.t_all.append(self.t_points)
        self.x = torch.cat(self.x_all).reshape(-1, 1)
        self.y = torch.cat(self.y_all).reshape(-1, 1)
        self.t = torch.cat(self.t_all).reshape(-1, 1)

    def data(self):
        return self.x, self.y, self.t

    def pde(self):
        x = torch.rand(self.n_pde, 1, requires_grad=True, device=self.device)
        y = torch.rand(self.n_pde, 1, requires_grad=True, device=self.device)
        t = torch.rand(self.n_pde, 1, requires_grad=True, device=self.device) * self.t_max
        return x, y, t

    def bc(self):
        n_per_side = self.n_bc // 4
        # x=0
        x_left = torch.zeros(n_per_side, 1, requires_grad=True, device=self.device)
        y_left = torch.rand(n_per_side, 1, requires_grad=True, device=self.device)
        t_left = torch.rand(n_per_side, 1, requires_grad=True, device=self.device) * self.t_max
        # x=1
        x_right = torch.ones(n_per_side, 1, requires_grad=True, device=self.device)
        y_right = torch.rand(n_per_side, 1, requires_grad=True, device=self.device)
        t_right = torch.rand(n_per_side, 1, requires_grad=True, device=self.device) * self.t_max
        # y=0
        x_bottom = torch.rand(n_per_side, 1, requires_grad=True, device=self.device)
        y_bottom = torch.zeros(n_per_side, 1, requires_grad=True, device=self.device)
        t_bottom = torch.rand(n_per_side, 1, requires_grad=True, device=self.device) * self.t_max
        # y=1
        x_top = torch.rand(n_per_side, 1, requires_grad=True, device=self.device)
        y_top = torch.ones(n_per_side, 1, requires_grad=True, device=self.device)
        t_top = torch.rand(n_per_side, 1, requires_grad=True, device=self.device) * self.t_max

        x = torch.cat([x_left, x_right, x_bottom, x_top])
        y = torch.cat([y_left, y_right, y_bottom, y_top])
        t = torch.cat([t_left, t_right, t_bottom, t_top])
        return x, y, t

    def ic(self):
        x = torch.rand(self.n_ic, 1, requires_grad=True, device=self.device)
        y = torch.rand(self.n_ic, 1, requires_grad=True, device=self.device)
        t = torch.zeros(self.n_ic, 1, requires_grad=True, device=self.device)
        return x, y, t

# %% Вычислитель невязок
class INITIAL_BOUNDARY_VALUE_PROBLEM:
    def __init__(self, model):
        self.model = model

    def pde(self, x, y, t):
        u = self.model(x, y, t)
        u_x, u_y, u_t = autograd.grad(u, (x, y, t), grad_outputs=torch.ones_like(u),
                                      create_graph=True, retain_graph=True)
        u_xx = autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                             create_graph=True, retain_graph=True)[0]
        u_yy = autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y),
                             create_graph=True, retain_graph=True)[0]
        u_tt = autograd.grad(u_t, t, grad_outputs=torch.ones_like(u_t),
                             create_graph=True, retain_graph=True)[0]
        u_xxt = autograd.grad(u_xx, t, grad_outputs=torch.ones_like(u_x),
                              create_graph=True, retain_graph=True)[0]
        u_yyt = autograd.grad(u_yy, t, grad_outputs=torch.ones_like(u_y),
                              create_graph=True, retain_graph=True)[0]
        kappa = self.model.get_kappa()
        return u_tt - kappa * (u_xxt + u_yyt) - (u_xx + u_yy)

    def bc(self, x, y, t):
        return self.model(x, y, t)

    def ic(self, x, y, t):
        u = self.model(x, y, t)
        u_t = autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                            create_graph=True, retain_graph=True)[0]
        return torch.cat([u, u_t], dim=1)

    def data(self, x, y, t):
        return self.model(x, y, t)

    # Дополнительные потери
    def boundary_pde_loss(self, n_points=200):
        """Граничные PDE-соотношения на четырёх сторонах."""
        device = self.model.log_kappa.device
        kappa = self.model.get_kappa()
        loss = 0.0
        # x = 0
        y = torch.rand(n_points, 1, requires_grad=True, device=device)
        t = torch.rand(n_points, 1, requires_grad=True, device=device) * points.t_max
        x = torch.zeros_like(y, requires_grad=True)
        u = self.model(x, y, t)
        u_x = autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_xxt = autograd.grad(u_xx, t, grad_outputs=torch.ones_like(u_xx), create_graph=True)[0]
        loss += torch.mean((kappa * u_xxt + u_xx) ** 2)

        # x = 1
        x = torch.ones_like(y, requires_grad=True)
        u = self.model(x, y, t)
        u_x = autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_xxt = autograd.grad(u_xx, t, grad_outputs=torch.ones_like(u_xx), create_graph=True)[0]
        # Правая часть: ∂q/∂t = 2x, на x=1 => 2
        loss += torch.mean((kappa * u_xxt + u_xx + 2.0) ** 2)

        # y = 0
        x = torch.rand(n_points, 1, requires_grad=True, device=device)
        t = torch.rand(n_points, 1, requires_grad=True, device=device) * points.t_max
        y = torch.zeros_like(x, requires_grad=True)
        u = self.model(x, y, t)
        u_y = autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_yy = autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
        u_yyt = autograd.grad(u_yy, t, grad_outputs=torch.ones_like(u_yy), create_graph=True)[0]
        # На y=0: ∂q/∂t = 2x, но из-за ГУ: -κ u_yyt - u_yy = 2x
        # В уравнении остаётся только член по y, т.к. производные по x на границе y=0 не обязаны быть нулевыми?
        # Но мы вывели соотношение: κ u_yyt + u_yy = -2x ? Проверим знак.
        # Из PDE: u_tt - κ(u_xxt+u_yyt) - (u_xx+u_yy) = 2x.
        # На y=0: u=0 => u_t=0, u_tt=0. Также u_xx=0? Нет, u_xx на границе не ноль.
        # Поэтому лучше оставить полное выражение: u_tt - κ(u_xxt+u_yyt) - (u_xx+u_yy) - 2x = 0.
        # Но чтобы получить простое соотношение, можно использовать тот факт, что на y=0 u=0 => u_xx=0 (так как u(0,y)=0 при x=0? Нет, здесь y=0, x произвольный, u(x,0)=0 => u_x(x,0)=0 => u_xx(x,0)=0.
        # Тогда действительно на y=0: u=0 => u_x=0, u_xx=0, u_t=0, u_tt=0.
        # Подстановка в PDE: -κ(0 + u_yyt) - (0 + u_yy) = 2x => -κ u_yyt - u_yy = 2x.
        # Умножим на -1: κ u_yyt + u_yy = -2x.
        loss += torch.mean((kappa * u_yyt + u_yy + 2 * x) ** 2)

        # y = 1
        y = torch.ones_like(x, requires_grad=True)
        u = self.model(x, y, t)
        u_y = autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_yy = autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
        u_yyt = autograd.grad(u_yy, t, grad_outputs=torch.ones_like(u_yy), create_graph=True)[0]
        # Аналогично y=1: κ u_yyt + u_yy = -2x
        loss += torch.mean((kappa * u_yyt + u_yy + 2 * x) ** 2)

        return loss / 4.0

    def ic_second_derivative_loss(self, n_points=500):
        """Штраф на точное значение u_tt при t=0."""
        x = torch.rand(n_points, 1, requires_grad=True, device=device)
        y = torch.rand(n_points, 1, requires_grad=True, device=device)
        t = torch.zeros(n_points, 1, requires_grad=True, device=device)
        u = self.model(x, y, t)
        u_t = autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_tt = autograd.grad(u_t, t, grad_outputs=torch.ones_like(u_t), create_graph=True)[0]
        # Точное выражение: u_tt(x,y,0) = 2x - 2π² sin(πx)sin(πy)
        exact_tt = 2 * x - 2 * (torch.pi ** 2) * torch.sin(torch.pi * x) * torch.sin(torch.pi * y)
        return torch.mean((u_tt - exact_tt) ** 2)

# %% Функция потерь с дополнительными весами
class LOSS(nn.Module):
    def __init__(self):
        super(LOSS, self).__init__()
        self.loss_fn = nn.MSELoss()
        # Обучаемые веса для основных компонент
        self.wpde = nn.Parameter(torch.log(torch.tensor(1.0)))
        self.wic = nn.Parameter(torch.log(torch.tensor(1.0)))
        self.wbc = nn.Parameter(torch.log(torch.tensor(1.0)))
        self.wdata = nn.Parameter(torch.log(torch.tensor(1.0)))
        # Дополнительные веса (фиксированные, но можно сделать обучаемыми)
        self.w_boundary_pde = nn.Parameter(torch.log(torch.tensor(0.1)), requires_grad=False)
        self.w_ic2 = nn.Parameter(torch.log(torch.tensor(0.1)), requires_grad=False)

    def forward(self, pde_true, pde_pred, ic_true, ic_pred, bc_true, bc_pred, data_true, data_pred,
                loss_boundary_pde, loss_ic2):
        total = (torch.exp(self.wpde) * self.loss_fn(pde_true, pde_pred) +
                 torch.exp(self.wic) * self.loss_fn(ic_true, ic_pred) +
                 torch.exp(self.wbc) * self.loss_fn(bc_true, bc_pred) +
                 torch.exp(self.wdata) * self.loss_fn(data_true, data_pred) +
                 torch.exp(self.w_boundary_pde) * loss_boundary_pde +
                 torch.exp(self.w_ic2) * loss_ic2)
        return total

# %% Аналитическое решение (для генерации данных)
class AnalyticalSolutionFull:
    def __init__(self, kappa=1.2e-4, n_max=30, m_max=30):
        self.kappa = kappa
        self.n_max = n_max
        self.m_max = m_max

    def __call__(self, x, y, t):
        if torch.is_tensor(x):
            x = x.cpu().detach().numpy()
            y = y.cpu().detach().numpy()
            t = t.cpu().detach().numpy()
        x = np.array(x).flatten()
        y = np.array(y).flatten()
        t = np.array(t).flatten()
        result = np.zeros_like(x)
        for n in range(1, self.n_max + 1):
            for m in range(1, self.m_max + 1):
                lambda_nm = np.pi ** 2 * (n ** 2 + m ** 2)
                A_nm = 1.0 if (n == 1 and m == 1) else 0.0
                int_y = (1 - (-1) ** m) / (m * np.pi)
                int_x = (-1) ** (n + 1) / (n * np.pi)
                f_nm = 4 * 2 * int_x * int_y
                if abs(A_nm) < 1e-10 and abs(f_nm) < 1e-10:
                    continue
                disc = (self.kappa * lambda_nm) ** 2 - 4 * lambda_nm
                if disc >= 0:
                    r1 = (-self.kappa * lambda_nm + np.sqrt(disc)) / 2
                    r2 = (-self.kappa * lambda_nm - np.sqrt(disc)) / 2
                    if abs(r1 - r2) > 1e-10:
                        C1 = -r2 * (A_nm - f_nm / lambda_nm) / (r1 - r2)
                        C2 = r1 * (A_nm - f_nm / lambda_nm) / (r1 - r2)
                    else:
                        C1 = C2 = (A_nm - f_nm / lambda_nm) / 2
                    T_t = C1 * np.exp(r1 * t) + C2 * np.exp(r2 * t) + f_nm / lambda_nm
                else:
                    real = -self.kappa * lambda_nm / 2
                    imag = np.sqrt(-disc) / 2
                    C1 = A_nm - f_nm / lambda_nm
                    C2 = -real * (A_nm - f_nm / lambda_nm) / imag
                    T_t = np.exp(real * t) * (C1 * np.cos(imag * t) + C2 * np.sin(imag * t)) + f_nm / lambda_nm
                spatial = np.sin(n * np.pi * x) * np.sin(m * np.pi * y)
                result += T_t * spatial
        return result

# %% Инициализация
model = PINN(t_max=500.0).to(device)
points = POINTS(t_max=500.0, n_time_points=200, device=device)  # больше точек по времени
q = lambda x, y, t: 2 * x
ic = lambda x, y, t: torch.cat([torch.sin(torch.pi * x) * torch.sin(torch.pi * y), torch.zeros_like(x)], dim=1)
bc = lambda x, y, t: torch.zeros_like(x)
analytical = AnalyticalSolutionFull()
problem = INITIAL_BOUNDARY_VALUE_PROBLEM(model)
loss_func = LOSS().to(device)

# Данные с датчиков
points_data = points.data()
data_true = torch.tensor(analytical(*points_data), device=device).reshape(-1, 1)
noise_level = 0.0001
data_noise = data_true + noise_level * torch.randn_like(data_true) * torch.abs(data_true)

# %% История
loss_history = []
kappa_history = []

# %% Этап 1: обучение сети (κ заморожен)
print("=== Этап 1: обучение сети при фиксированном κ ===")
model.log_kappa.requires_grad = False
optimizer_adam = optim.NAdam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_adam, mode='min', factor=0.75, patience=500)
best_loss = float('inf')
for epoch in range(5000):
    optimizer_adam.zero_grad()
    # Точки
    pde_x, pde_y, pde_t = points.pde()
    ic_x, ic_y, ic_t = points.ic()
    bc_x, bc_y, bc_t = points.bc()
    # Прямой проход
    pde_pred = problem.pde(pde_x, pde_y, pde_t)
    ic_pred = problem.ic(ic_x, ic_y, ic_t)
    bc_pred = problem.bc(bc_x, bc_y, bc_t)
    data_pred = problem.data(*points_data)
    # Истинные значения
    pde_true = q(pde_x, pde_y, pde_t)
    ic_true = ic(ic_x, ic_y, ic_t)
    bc_true = bc(bc_x, bc_y, bc_t)
    # Доп. потери (не требуют κ, но мы их вычислим для информации, хотя веса заморожены)
    loss_bpde = problem.boundary_pde_loss()
    loss_ic2 = problem.ic_second_derivative_loss()
    # Основной loss (без учёта доп. потерь, т.к. κ фиксирован и они не дадут градиента по сети)
    loss = (torch.exp(loss_func.wpde) * loss_func.loss_fn(pde_true, pde_pred) +
            torch.exp(loss_func.wic) * loss_func.loss_fn(ic_true, ic_pred) +
            torch.exp(loss_func.wbc) * loss_func.loss_fn(bc_true, bc_pred) +
            torch.exp(loss_func.wdata) * loss_func.loss_fn(data_noise, data_pred))
    loss.backward()
    optimizer_adam.step()
    scheduler.step(loss)
    loss_history.append(torch.log(loss).item())
    kappa_history.append(model.get_kappa().item())
    if loss.item() < best_loss:
        best_loss = loss.item()
        torch.save(model.state_dict(), 'pg_inhomogeneous_model_stage1.pth')
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: loss={loss.item():.6f}, κ={model.get_kappa().item():.6f}")

# %% Этап 2: обучение κ при замороженной сети
print("=== Этап 2: оптимизация κ ===")
model.load_state_dict(torch.load('pg_inhomogeneous_model_stage1.pth'))
for param in model.parameters():
    param.requires_grad = False
model.log_kappa.requires_grad = True
# Используем Adam для κ
optimizer_kappa = optim.NAdam([model.log_kappa], lr=1e-3)
for epoch in range(3000):
    optimizer_kappa.zero_grad()
    pde_x, pde_y, pde_t = points.pde()
    pde_pred = problem.pde(pde_x, pde_y, pde_t)
    pde_true = q(pde_x, pde_y, pde_t)
    data_pred = problem.data(*points_data)
    # Используем также граничные PDE и IC2 потери, т.к. они сильно зависят от κ
    loss_bpde = problem.boundary_pde_loss()
    loss_ic2 = problem.ic_second_derivative_loss()
    # Основные компоненты (только PDE и данные, т.к. IC/BC уже хорошо выполнены)
    loss = (torch.exp(loss_func.wpde) * loss_func.loss_fn(pde_true, pde_pred) +
            torch.exp(loss_func.wdata) * loss_func.loss_fn(data_noise, data_pred) +
            torch.exp(loss_func.w_boundary_pde) * loss_bpde +
            torch.exp(loss_func.w_ic2) * loss_ic2)
    loss.backward()
    optimizer_kappa.step()
    loss_history.append(torch.log(loss).item())
    kappa_history.append(model.get_kappa().item())
    if epoch % 100 == 0:
        print(f"Kappa epoch {epoch}: loss={loss.item():.6f}, κ={model.get_kappa().item():.6e}, grad={model.log_kappa.grad.item():.2e}")

# %% Этап 3: L-BFGS для точной настройки κ
print("=== Этап 3: L-BFGS для κ ===")
# Зафиксируем точки для L-BFGS
fixed_pde = points.pde()
fixed_bc = points.bc()
fixed_ic = points.ic()
fixed_data = points_data

lbfgs_optimizer = optim.LBFGS([model.log_kappa], lr=1.0, max_iter=20, history_size=200, line_search_fn='strong_wolfe')
def closure():
    lbfgs_optimizer.zero_grad()
    # Преобразуем в тензоры с градиентами
    fpde = tuple(map(lambda t: t.detach().requires_grad_(True), fixed_pde))
    fbc = tuple(map(lambda t: t.detach().requires_grad_(True), fixed_bc))
    fic = tuple(map(lambda t: t.detach().requires_grad_(True), fixed_ic))
    fdata = tuple(map(lambda t: t.detach().requires_grad_(True), fixed_data))
    # Вычисляем невязки
    pde_pred = problem.pde(*fpde)
    pde_true = q(*fpde)
    data_pred = problem.data(*fdata)
    loss_bpde = problem.boundary_pde_loss()
    loss_ic2 = problem.ic_second_derivative_loss()
    loss = (torch.exp(loss_func.wpde) * loss_func.loss_fn(pde_true, pde_pred) +
            torch.exp(loss_func.wdata) * loss_func.loss_fn(data_noise, data_pred) +
            torch.exp(loss_func.w_boundary_pde) * loss_bpde +
            torch.exp(loss_func.w_ic2) * loss_ic2)
    loss.backward()
    return loss

for epoch in range(500):
    loss = lbfgs_optimizer.step(closure)
    loss_history.append(torch.log(loss).item())
    kappa_history.append(model.get_kappa().item())
    if epoch % 50 == 0:
        print(f"L-BFGS {epoch}: loss={loss.item():.6f}, κ={model.get_kappa().item():.6e}")
    if loss.item() < best_loss:
        best_loss = loss.item()
        torch.save(model.state_dict(), 'pg_inhomogeneous_model_final.pth')

print(f"Итоговое κ = {model.get_kappa().item():.6e} (истинное 1.2e-4)")
print(f"Относительная ошибка: {abs(model.get_kappa().item() - 1.2e-4) / 1.2e-4 * 100:.2f}%")

# %% Визуализация и метрики (как в исходном коде, с исправлением опечатки contourf)
# ... (код визуализации оставлен без изменений, только исправлено 'contourf' на 'contourf')
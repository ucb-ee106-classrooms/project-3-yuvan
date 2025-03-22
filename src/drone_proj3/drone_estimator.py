import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.family'] = ['Arial']
plt.rcParams['font.size'] = 14
import time

class Estimator:
    """A base class to represent an estimator.

    This module contains the basic elements of an estimator, on which the
    subsequent DeadReckoning, Kalman Filter, and Extended Kalman Filter classes
    will be based on. A plotting function is provided to visualize the
    estimation results in real time.

    Attributes:
    ----------
        u : list
            A list of system inputs, where, for the ith data point u[i],
            u[i][1] is the thrust of the quadrotor
            u[i][2] is right wheel rotational speed (rad/s).
        x : list
            A list of system states, where, for the ith data point x[i],
            x[i][0] is translational position in x (m),
            x[i][1] is translational position in z (m),
            x[i][2] is the bearing (rad) of the quadrotor
            x[i][3] is translational velocity in x (m/s),
            x[i][4] is translational velocity in z (m/s),
            x[i][5] is angular velocity (rad/s),
        y : list
            A list of system outputs, where, for the ith data point y[i],
            y[i][1] is distance to the landmark (m)
            y[i][2] is relative bearing (rad) w.r.t. the landmark
        x_hat : list
            A list of estimated system states. It should follow the same format
            as x.
        dt : float
            Update frequency of the estimator.
        fig : Figure
            matplotlib Figure for real-time plotting.
        axd : dict
            A dictionary of matplotlib Axis for real-time plotting.
        ln* : Line
            matplotlib Line object for ground truth states.
        ln_*_hat : Line
            matplotlib Line object for estimated states.
        canvas_title : str
            Title of the real-time plot, which is chosen to be estimator type.

    Notes
    ----------
        The landmark is positioned at (0, 5, 5).
    """
    # noinspection PyTypeChecker
    def __init__(self, is_noisy=False):
        self.u = []
        self.x = []
        self.y = []
        self.computation_times = []
        self.x_hat = []  # Your estimates go here!
        self.t = []
        self.fig, self.axd = plt.subplot_mosaic(
            [['xz', 'phi'],
             ['xz', 'x'],
             ['xz', 'z'],
             ['xy', 'time']], figsize=(20.0, 10.0))
        self.ln_xz, = self.axd['xz'].plot([], 'o-g', linewidth=2, label='True')
        self.ln_xz_hat, = self.axd['xz'].plot([], 'o-c', label='Estimated')
        self.ln_phi, = self.axd['phi'].plot([], 'o-g', linewidth=2, label='True')
        self.ln_phi_hat, = self.axd['phi'].plot([], 'o-c', label='Estimated')
        self.ln_x, = self.axd['x'].plot([], 'o-g', linewidth=2, label='True')
        self.ln_x_hat, = self.axd['x'].plot([], 'o-c', label='Estimated')
        self.ln_z, = self.axd['z'].plot([], 'o-g', linewidth=2, label='True')
        self.ln_z_hat, = self.axd['z'].plot([], 'o-c', label='Estimated')
        self.ln_time, = self.axd['time'].plot([], 'o-m', label='Computation Time')
        self.axd['time'].set_ylabel('Computation Time (s)')
        self.axd['time'].set_xlabel('Iteration')
        self.axd['time'].legend()
        self.canvas_title = 'N/A'

        # Defined in dynamics.py for the dynamics model
        # m is the mass and J is the moment of inertia of the quadrotor 
        self.gr = 9.81 
        self.m = 0.92
        self.J = 0.0023
        # These are the X, Y, Z coordinates of the landmark
        self.landmark = (0, 5, 5)

        # This is a (N,12) where it's time, x, u, then y_obs 
        if is_noisy:
            with open('noisy_data.npy', 'rb') as f:
                self.data = np.load(f)
        else:
            with open('data.npy', 'rb') as f:
                self.data = np.load(f)

        self.dt = self.data[-1][0]/self.data.shape[0]


    def run(self):
        for i, data in enumerate(self.data):
            self.t.append(np.array(data[0]))
            self.x.append(np.array(data[1:7]))
            self.u.append(np.array(data[7:9]))
            self.y.append(np.array(data[9:12]))
            if i == 0:
                self.x_hat.append(self.x[-1])
            else:
                self.update(i)
        print(f"computation time average is: {sum(self.computation_times)/len(self.computation_times)}")
        print(f"average error in x_hat vs x is: {np.mean(np.abs(np.array(self.x_hat) - np.array(self.x)), axis=0)}")
        return self.x_hat

    def update(self, _):
        raise NotImplementedError

    def plot_init(self):
        self.axd['xz'].set_title(self.canvas_title)
        self.axd['xz'].set_xlabel('x (m)')
        self.axd['xz'].set_ylabel('z (m)')
        self.axd['xz'].set_aspect('equal', adjustable='box')
        self.axd['xz'].legend()
        self.axd['phi'].set_ylabel('phi (rad)')
        self.axd['phi'].set_xlabel('t (s)')
        self.axd['phi'].legend()
        self.axd['x'].set_ylabel('x (m)')
        self.axd['x'].set_xlabel('t (s)')
        self.axd['x'].legend()
        self.axd['z'].set_ylabel('z (m)')
        self.axd['z'].set_xlabel('t (s)')
        self.axd['z'].legend()
        plt.tight_layout()
    
    def plot_time(self, ln, data):
        if len(data):
            x = list(range(len(data)))
            ln.set_data(x, data)
            self.axd['time'].set_xlim([0, max(x) + 1])
            self.axd['time'].set_ylim([0, max(data) * 1.1])


    def plot_update(self, _):
        self.plot_xzline(self.ln_xz, self.x)
        self.plot_xzline(self.ln_xz_hat, self.x_hat)
        self.plot_philine(self.ln_phi, self.x)
        self.plot_philine(self.ln_phi_hat, self.x_hat)
        self.plot_xline(self.ln_x, self.x)
        self.plot_xline(self.ln_x_hat, self.x_hat)
        self.plot_zline(self.ln_z, self.x)
        self.plot_zline(self.ln_z_hat, self.x_hat)
        self.plot_time(self.ln_time, self.computation_times)


    def plot_xzline(self, ln, data):
        if len(data):
            x = [d[0] for d in data]
            z = [d[1] for d in data]
            ln.set_data(x, z)
            self.resize_lim(self.axd['xz'], x, z)

    def plot_philine(self, ln, data):
        if len(data):
            t = self.t
            phi = [d[2] for d in data]
            ln.set_data(t, phi)
            self.resize_lim(self.axd['phi'], t, phi)

    def plot_xline(self, ln, data):
        if len(data):
            t = self.t
            x = [d[0] for d in data]
            ln.set_data(t, x)
            self.resize_lim(self.axd['x'], t, x)

    def plot_zline(self, ln, data):
        if len(data):
            t = self.t
            z = [d[1] for d in data]
            ln.set_data(t, z)
            self.resize_lim(self.axd['z'], t, z)

    # noinspection PyMethodMayBeStatic
    def resize_lim(self, ax, x, y):
        xlim = ax.get_xlim()
        ax.set_xlim([min(min(x) * 1.05, xlim[0]), max(max(x) * 1.05, xlim[1])])
        ylim = ax.get_ylim()
        ax.set_ylim([min(min(y) * 1.05, ylim[0]), max(max(y) * 1.05, ylim[1])])

class OracleObserver(Estimator):
    """Oracle observer which has access to the true state.

    This class is intended as a bare minimum example for you to understand how
    to work with the code.

    Example
    ----------
    To run the oracle observer:
        $ python drone_estimator_node.py --estimator oracle_observer
    """
    def __init__(self, is_noisy=False):
        super().__init__(is_noisy)
        self.canvas_title = 'Oracle Observer'

    def update(self, _):
        self.x_hat.append(self.x[-1])


class DeadReckoning(Estimator):
    """Dead reckoning estimator.

    Your task is to implement the update method of this class using only the
    u attribute and x0. You will need to build a model of the unicycle model
    with the parameters provided to you in the lab doc. After building the
    model, use the provided inputs to estimate system state over time.

    The method should closely predict the state evolution if the system is
    free of noise. You may use this knowledge to verify your implementation.

    Example
    ----------
    To run dead reckoning:
        $ python drone_estimator_node.py --estimator dead_reckoning
    """
    def __init__(self, is_noisy=False):
        super().__init__(is_noisy)
        self.canvas_title = 'Dead Reckoning'

    def update(self, _):
        if len(self.x_hat) == 0:
            return

        if len(self.u) == 0:
            return
        start_time = time.time()

        x_prev, z_prev, phi_prev, x_dot_prev, z_dot_prev, phi_dot_prev = self.x_hat[-1]

        u_1, u_2 = self.u[-1][0], self.u[-1][1]
        x_new = x_prev + x_dot_prev * self.dt
        z_new = z_prev + z_dot_prev * self.dt
        phi_new = phi_prev + phi_dot_prev * self.dt

        x_dot_new = x_dot_prev - np.sin(phi_prev) * u_1 * self.dt/self.m
        z_dot_new = z_dot_prev + (-self.gr + (np.cos(phi_prev)/self.m) * u_1) * self.dt
        phi_dot_new = phi_dot_prev + u_2 * self.dt / self.J

        self.x_hat.append([x_new, z_new, phi_new, x_dot_new, z_dot_new, phi_dot_new])
        self.computation_times.append(time.time() - start_time)


# noinspection PyPep8Naming
class ExtendedKalmanFilter(Estimator):
    """Extended Kalman filter estimator.

    Your task is to implement the update method of this class using the u
    attribute, y attribute, and x0. You will need to build a model of the
    unicycle model and linearize it at every operating point. After building the
    model, use the provided inputs and outputs to estimate system state over
    time via the recursive extended Kalman filter update rule.

    Hint: You may want to reuse your code from DeadReckoning class and
    KalmanFilter class.

    Attributes:
    ----------
        landmark : tuple
            A tuple of the coordinates of the landmark.
            landmark[0] is the x coordinate.
            landmark[1] is the y coordinate.
            landmark[2] is the z coordinate.

    Example
    ----------
    To run the extended Kalman filter:
        $ python drone_estimator_node.py --estimator extended_kalman_filter
    """
    def __init__(self, is_noisy=False):
        super().__init__(is_noisy)
        self.canvas_title = 'Extended Kalman Filter'
        # TODO: Your implementation goes here!
        # You may define the Q, R, and P matrices below.
        self.A = np.zeros((6, 6))
        self.A[0, 3] = 1
        self.A[1, 4] = 1
        self.A[2, 5] = 1
        self.B = None
        self.C = np.zeros((2, 6))
        self.C[1, 2] = 1
        self.Q = np.eye(6)
        self.R = np.eye(2)
        self.P = np.eye(6)

    def update(self, i):
        if len(self.x_hat) > 0:
            start_time = time.time()
            # TODO: Your implementation goes here!
            # You may use self.u, self.y, and self.x[0] for estimation
            naive_x_hat = self.g(self.x_hat[-1], self.u[-1])
            self.approx_A(self.x_hat[-1], self.u[-1])
            conditional_P = self.A @ self.P @ self.A.T + self.Q
            self.approx_C(naive_x_hat)
            K = conditional_P @ self.C.T @ np.linalg.inv(self.C @ conditional_P @ self.C.T + self.R)
            self.x_hat.append(naive_x_hat + K @ (self.y[-1] - self.h(naive_x_hat, self.y[-1])))
            self.P = (np.eye(6) - K @ self.C) @ conditional_P
            self.computation_times.append(time.time() - start_time)
            print(self.computation_times[-1])

            


    def g(self, x, u):
        x_prev, z_prev, phi_prev, x_dot_prev, z_dot_prev, phi_dot_prev = x

        u_1, u_2 = u
        x_new = x_prev + x_dot_prev * self.dt
        z_new = z_prev + z_dot_prev * self.dt
        phi_new = phi_prev + phi_dot_prev * self.dt

        x_dot_new = x_dot_prev - np.sin(phi_prev) * u_1 * self.dt/self.m
        z_dot_new = z_dot_prev + (-self.gr + (np.cos(phi_prev)/self.m) * u_1) * self.dt
        phi_dot_new = phi_dot_prev + u_2 * self.dt / self.J
        return [x_new, z_new, phi_new, x_dot_new, z_dot_new, phi_dot_new]

    def h(self, x, y_obs):
        num_1 = np.sqrt((x[0] - self.landmark[0])**2 + self.landmark[1] ** 2 + (x[1] - self.landmark[2])**2)
        num_2 = x[2]
        return np.array([num_1, num_2])

    def approx_A(self, x, u):
        self.A[3, 2] = -np.cos(x[2]) * u[0] / self.m
        self.A[4, 2] = -np.sin(x[2]) * u[0] / self.m
    
    def approx_C(self, x):
        denominator = np.sqrt((x[0] - self.landmark[0])**2 + self.landmark[1] ** 2 + (x[1] - self.landmark[2])**2)
        self.C[0, 0] = (x[0] - self.landmark[0]) / denominator
        self.C[0, 1] = (x[1] - self.landmark[2]) / denominator

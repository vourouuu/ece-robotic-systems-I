import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.transforms as transforms

'''
Covariences
    - P    : covariance for noise of motion model
    - R    : covariance for noise of measurement model
    - R_SIM: covariance for noise of true state
'''

''' Noise ---> 0
    P = np.diag([0., 0., np.deg2rad(0)]) ** 2
    R = np.diag([0., np.deg2rad(0)]) ** 2
    R_SIM = np.diag([0., 0., np.deg2rad(0)]) ** 2
'''

''' Noise ---> normal '''
P = np.diag([0.1, 0.1, np.deg2rad(2)]) ** 2
R = np.diag([0.3, np.deg2rad(5)]) ** 2
R_SIM = np.diag([0.01, 0.01, np.deg2rad(1)]) ** 2

''' Noise ---> high
    P = np.diag([0.8, 0.8, np.deg2rad(6)]) ** 2
    R = np.diag([0.5, np.deg2rad(10)]) ** 2
    R_SIM = np.diag([0.01, 0.01, np.deg2rad(1)]) ** 2
'''

'''
Constants
    - M            : number of particles
    - STEPS        : number of simulation steps
    - SENSOR_RANGE : LiDAR range
    - RADIUS       : wheel radii
    - DISTANCE     : the distance of the wheels from the robot center
    - dt           : time step
'''
M = 50
STEPS = 100
SENSOR_RANGE = 12.0
RADIUS = 0.2
DISTANCE = 0.5
dt = 1.0

class LandmarkEKF:
    def __init__(self, mu, Sigma, tau=1.0):
        self.mu = np.array(mu)       # Mean position [x, y]
        self.Sigma = np.array(Sigma) # Covariance (2x2)
        self.tau = tau               # Existence probability

class Particle:
    def __init__(self, state):
        self.x   = state[0]
        self.y   = state[1]
        self.theta = state[2]
        self.weight = 1.0    
        self.landmarks = [] # List of LandmarkEKF

def normalize_angle(angle):
    while (angle > np.pi): angle -= 2 * np.pi
    while (angle < -np.pi): angle += 2 * np.pi
    
    return angle

def control_input():
    '''
    Generates random control inputs for the robot's motors:
        - uL: random velocity command for left motor
        - uR: random velocity command for right motor
    '''
    uL = 1.0 + np.random.randn() * 1.5  
    uR = 1.2 + np.random.randn() * 1.5
    u  = [uL, uR]
    
    return u

def motion_model(state, u, dt):
    '''
    Computes the robot state using control input (u), previous state (state) and time step (dt)
    via Kinematic Equation and Euler Integration:
        - RADIUS   : wheel radii
        - DISTANCE : the distance of the wheels from the robot center
        - theta    : angle of robot state
        - uL       : left wheel angular velocity
        - uR       : right wheel angular velocity
        - state_dot: kinematic equation
    '''
    _, _, theta = state
    state = state.reshape(3,1)
    theta = normalize_angle(theta)
    uL, uR = u
    
    A = np.array([[(RADIUS/2)*np.cos(theta),   (RADIUS/2)*np.cos(theta)],
                  [(RADIUS/2)*np.sin(theta),   (RADIUS/2)*np.sin(theta)],
                  [    -RADIUS/(2*DISTANCE),        RADIUS/(2*DISTANCE)]])        
    B = np.array([[uL], [uR]])
    
    state_dot = A @ B 
    state += state_dot * dt # Euler Integration
    state[2] = normalize_angle(state[2])
    
    return state.reshape(3,1)

def measurement_model(landmark, state):
    '''
    Measures the distance and angle between landmark and robot state.
        - r  : distance
        - phi: angle
    '''
    dx = landmark[0] - state[0]
    dy = landmark[1] - state[1]
    r = np.sqrt(dx**2 + dy**2)
    phi = normalize_angle(np.arctan2(dy, dx) - state[2])
    
    return np.array([r, phi]).reshape(2, 1)

def measurement_model_inverse(z, state):
    '''
    Computes the landmark state using robot state (state) and measuremnt (z).
    '''
    sx = state[0]
    sy = state[1]
    theta = normalize_angle(state[2])

    r = z[0]
    phi = normalize_angle(z[1])

    lx = sx + r * np.cos(normalize_angle(theta + phi))
    ly = sy + r * np.sin(normalize_angle(theta + phi))

    return np.array([lx,ly]).reshape(2, 1)

def sampling_state(mu, Sigma):
    return np.random.multivariate_normal(mu, Sigma)

def fastslam2(particles, u: list[float], z:list[tuple[int, np.ndarray]],
                   dt: float, P, R, rho_plus=2.2, rho_minus=0.72, p0=0.1):
    S_aux = []
    for m in range(M):
        '''
        Temporary lists
            - state_p_temp : contains particle state estimations based on each measurement 
            - w_p_temp     : contains particle weights based on each measurement
            - n_hat_list   : contains beliefs of landmark IDs based on each measurement
            - new_lm_p_temp: contains every new landmark to be appended into particle's list of landmarks
        '''
        
        state_p_temp = []
        w_p_temp = []
        n_hat_list = []
        new_lm_p_temp = []

        p = deepcopy(particles[m]) # copy of m-th particle 
        
        '''
        If there are no landmarks, we still have to update the particle state and weight:
            - s_prev           : previous particle state
            - s_hat            : estimation of particle state based only on motion model
            - p.x, p.y, p.theta: updated particle state
            - p.weight         : updated particle weight
        Else:
            FastSLAM 2.0
        '''
        if len(z) == 0:
            s_prev = np.array([p.x, p.y, p.theta]) 
            s_hat = motion_model(s_prev, u, dt)
            p.x, p.y, p.theta = s_hat.flatten() + sampling_state(np.zeros((3,)), P) 
            p.theta = normalize_angle(p.theta)
            p.weight = p0
        else:
            for _, zi in z:
                '''
                Variables:
                    - zt      : i-th measurement
                    - N_prev  : number of landamrks the particle has seen
                    - s_prev  : previous particle state
                    - s_hat   : estimation of particle state based only on motion model
                    - id_ML   : ID of the landmark with the maximum likelihood
                    - s_ML    : prediction state based on the landmark with the maximum likelihood
                    - max_p_nt: maximum likelihood
                '''
                zt = zi.reshape(2, 1)

                N_prev = len(p.landmarks)
                s_prev = np.array([p.x, p.y, p.theta])
                s_hat = motion_model(s_prev, u, dt)

                id_ML = N_prev 
                s_ML = s_hat.flatten() + sampling_state(np.zeros((3,)), P)
                s_ML = s_ML.reshape(3, 1)
                max_p_nt = -1
                
                # 1. Extending the path posterior by sampling new poses
                for n in range(0, N_prev):
                    '''
                    Variables:
                        - (lx, ly): landmark position
                        - z_hat   : estimation of landmark position based on the g^{-1}
                        - G_theta : derivative of measurement model wrt landmark position
                        - G_s     : derivative of measurement model wrt particle state
                        - Sigma_st: covariance of Gaussian from which the new state will be sampled from
                        - mu_st   : mean of Gaussian from which the new state will be sampled from
                        - s       : new sampled state
                    '''
                    z_hat = measurement_model(p.landmarks[n].mu, s_hat)

                    lx = p.landmarks[n].mu[0, 0]
                    ly = p.landmarks[n].mu[1, 0]
                    sx = s_hat[0, 0]
                    sy = s_hat[1, 0]
                    dx = lx - sx
                    dy = ly - sy
                    r = np.sqrt(dx**2 + dy**2)

                    G_theta = np.array([[    dx/r,    dy/r], 
                                        [-dy/r**2, dx/r**2]])
                    G_s = np.array([[  -dx/r,    -dy/r,  0], 
                                    [dy/r**2, -dx/r**2, -1]])
                    
                    Q = R + G_theta @ p.landmarks[n].Sigma @ G_theta.T
                    Sigma_st = np.linalg.pinv(G_s.T @ np.linalg.pinv(Q) @ G_s + np.linalg.pinv(P))
                    mu_st = Sigma_st @ G_s.T @ np.linalg.pinv(Q) @ (zt - z_hat) + s_hat
                    s = sampling_state(mu_st.reshape(3,), Sigma_st)

                    '''
                    Computes the probability of the measurement to be associated with an already observed landmark.
                    '''
                    if np.linalg.det(Q) > 0:
                        p_nt = ((2 * np.pi * np.linalg.det(Q))**(-0.5)) * np.exp(-0.5 * (zt - measurement_model(p.landmarks[n].mu, s)).T @ np.linalg.pinv(Q) @ (zt - measurement_model(p.landmarks[n].mu, s)))
                    else:
                        p_nt = 1e-10
                    
                    '''
                    This way we keep the maximum likelihood landmark and the state prediction that corresponds to it.
                    '''
                    if (p_nt >= max_p_nt) and (p_nt > p0):
                        max_p_nt = p_nt
                        s_ML = s
                        id_ML = n

                state_p_temp.append(s_ML.reshape(3, 1))

                # 2. Data association
                '''
                Variables:
                    - n_hat: the landmark we guess we saw is the one with the greater probability p_nt
                '''
                n_hat = id_ML
                n_hat_list.append(n_hat)
                
                # 3. Process measurement
                for n in range(0, N_prev + 1):
                    # Known landmark
                    if (n == n_hat) and (n_hat <= N_prev - 1):
                        lx = p.landmarks[n].mu[0]
                        ly = p.landmarks[n].mu[1]
                        
                        sx = s_ML[0]
                        sy = s_ML[1]
                        
                        dx = lx - sx
                        dy = ly - sy
                        r = np.sqrt(dx**2 + dy**2)

                        G_theta = np.block([[    dx/r,    dy/r], 
                                            [-dy/r**2, dx/r**2]])
                        G_s = np.block([[  -dx/r,    -dy/r,  0], 
                                        [dy/r**2, -dx/r**2, -1]])
                        
                        Q = R + G_theta @ p.landmarks[n].Sigma @ G_theta.T

                        p.landmarks[n].tau = p.landmarks[n].tau + rho_plus # τ = τ + (ρ+)
                        K = p.landmarks[n].Sigma @ G_theta.T @ np.linalg.pinv(Q)
                        p.landmarks[n].mu = p.landmarks[n].mu + K @ (zt - measurement_model(p.landmarks[n].mu, s_ML))
                        L = G_s @ P @ G_s.T + G_theta @ p.landmarks[n].Sigma @ G_theta.T + R
                        p.landmarks[n].Sigma = (np.eye(2) - K @ G_theta) @ p.landmarks[n].Sigma
                        weight = (1/(np.sqrt(2 * np.pi * np.linalg.det(L)))) * np.exp(-0.5 * (zt - measurement_model(p.landmarks[n].mu, s_ML)).T @ np.linalg.pinv(L) @ (zt - measurement_model(p.landmarks[n].mu, s_ML)))
                        w_p_temp.append(weight[0, 0])
                    # New landmark
                    elif (n == n_hat) and (n_hat == N_prev):   
                        n = N_prev + 1
                        p.weight = p0
                        s_new_landmark = s_hat.flat + sampling_state(np.zeros((3,)), P)
                        mu_n = measurement_model_inverse(zt, s_new_landmark)
                        
                        lx = mu_n[0]
                        ly = mu_n[1]
                        sx = s_new_landmark[0]
                        sy = s_new_landmark[1]
                        dx = lx - sx
                        dy = ly - sy
                        r = np.sqrt(dx**2 + dy**2)

                        G_theta = np.array([[    dx/r,    dy/r], 
                                            [-dy/r**2, dx/r**2]]).reshape(2, 2)
                        Sigma_n = np.linalg.pinv(G_theta @ np.linalg.pinv(R) @ G_theta.T)
                        
                        new_lm_p_temp.append(LandmarkEKF(mu_n, Sigma_n, rho_plus))
                        w_p_temp.append(p.weight)
            
            # Updated particle
            p.x, p.y, p.theta = np.mean(np.block(state_p_temp), axis=1)
            p.theta = normalize_angle(p.theta)
            p.weight = np.mean(w_p_temp)
            
            # Handle Unobserved Landmarks (HUL)
            n_HUL = 0
            while(n_HUL < N_prev):
                if ((p.landmarks[n_HUL].mu[0] - s_ML[0])**2) + ((p.landmarks[n_HUL].mu[1] - s_ML[1])**2)  >  SENSOR_RANGE**2:
                    n_HUL += 1
                    pass
                elif n_HUL not in n_hat_list:
                    p.landmarks[n_HUL].tau = p.landmarks[n_HUL].tau - rho_minus
                    if p.landmarks[n_HUL].tau < 0:
                        del p.landmarks[n_HUL]
                        n_HUL -= 1
                        N_prev = len(p.landmarks)
                    else:
                        n_HUL += 1
                else:
                    n_HUL += 1

            for lm in new_lm_p_temp:
                p.landmarks.append(lm)

        # Update S_aux list with updated particle        
        S_aux.append(p)

    # Normalize and update weights
    weights = np.array([p.weight for p in S_aux])
    if np.sum(weights) > 0:
        weights = weights / np.sum(weights)
    else:
        weights = np.ones(M) / M

    # Resampling
    new_particles = []
    cumsum = np.cumsum(weights)
    for i in range(M):
        r = np.random.rand()
        idx = np.searchsorted(cumsum, r)
        new_particles.append(deepcopy(S_aux[idx]))
        new_particles[-1].weight = 1.0
        
    return new_particles

def calc_final_state(particles):
    '''
    Calculates the final state as the mean of the particle states 
    '''
    x = np.mean([p.x for p in particles])
    y = np.mean([p.y for p in particles])
    theta = np.mean([p.theta for p in particles])
    theta = normalize_angle(theta)

    return np.array([x, y, theta])

def main():
    true_landmarks = np.array([[ 1,  1], [-2.5, -2.5], 
                               [12,  3], [   3,    5],
                               [11.5, 18.5], [-20, -5]])
    
    x_true = np.zeros((1, 3)).flatten()

    # Initialize history as lists to avoid the 'append' error
    hist_est = []
    hist_true = []

    # Initialize particles
    particles = [Particle(x_true) for _ in range(M)]

    # Visualization setup
    plt.ion()
    fig, ax = plt.subplots(figsize=(30, 30))

    # Simulation loop
    for t in range(STEPS):
        u = control_input() 
        x_true = motion_model(x_true, u, dt).reshape(1, 3).flatten()
        x_true += sampling_state(np.zeros((3,)), R_SIM)

        measurements = []
        for i, lm in enumerate(true_landmarks):
            dx = lm[0] - x_true[0]
            dy = lm[1] - x_true[1]
            distance = np.sqrt(dx**2 + dy**2)

            if distance < SENSOR_RANGE:
                range_noise = np.random.normal(0, 0.05)
                bearing_noise = np.random.normal(0.05, np.deg2rad(1))
                range_obs = distance + range_noise
                bearing_obs = np.arctan2(dy, dx) - x_true[2] + bearing_noise
                bearing_obs = normalize_angle(bearing_obs)
                measurements.append((i, np.array([range_obs, bearing_obs])))

        # FastSLAM 2.0 step
        particles = fastslam2(particles, u, measurements, dt, P, R)

        # Store history as lists
        est_pose = calc_final_state(particles)
        hist_est.append(est_pose)
        hist_true.append(x_true.copy())

        # Visualization
        ax.clear()
        # Plot particles
        plotted_landmark_label = False

        for p in particles:
            ax.plot(p.x, p.y, '.r', alpha=0.2)
            for lm in p.landmarks:
                if not plotted_landmark_label:
                    ax.plot(lm.mu[0], lm.mu[1], 'bx', alpha=1, label="Estimated Landmarks")
                    plotted_landmark_label = True
                else:
                    ax.plot(lm.mu[0], lm.mu[1], 'bx', alpha=0.2)
        
        # Plot true landmarks
        ax.plot(true_landmarks[:, 0], true_landmarks[:, 1], 'k*', markersize=12, label='True Landmarks', alpha=0.3)
        
        # Plot true and estimated trajectory
        if len(hist_est) > 1:
            hist_est_arr = np.array(hist_est)
            hist_true_arr = np.array(hist_true)
            ax.plot(hist_true_arr[:, 0], hist_true_arr[:, 1], '-g', label='True Trajectory')
            ax.plot(hist_est_arr[:, 0], hist_est_arr[:, 1], '-r', label='Estimated Trajectory')
        
        rect_width = 0.5
        rect_height = 0.4
        line_length = 0.5

        # True state
        x, y, theta = x_true[0], x_true[1], x_true[2]
        rect_true = Rectangle((-rect_width/2, -rect_height/2), rect_width, rect_height,
                     edgecolor='g', facecolor='none', linewidth=1, zorder=10)
        t = transforms.Affine2D().rotate(theta).translate(x, y) + ax.transData
        rect_true.set_transform(t)
        ax.add_patch(rect_true)

        end_x = x + line_length * np.cos(theta)
        end_y = y + line_length * np.sin(theta)
        ax.plot([x, end_x], [y, end_y], 'g-', linewidth=2, zorder=11)
        ax.plot(x_true[0], x_true[1], 'go', markersize=5, label='True Pose')

        # Estimated state
        x, y, theta = est_pose[0], est_pose[1], est_pose[2]
        rect_est = Rectangle((-rect_width/2, -rect_height/2), rect_width, rect_height,
                     edgecolor='r', facecolor='none', linewidth=1, zorder=10)
        t = transforms.Affine2D().rotate(theta).translate(x, y) + ax.transData
        rect_est.set_transform(t)
        ax.add_patch(rect_est)

        end_x = x + line_length * np.cos(theta)
        end_y = y + line_length * np.sin(theta)
        ax.plot([x, end_x], [y, end_y], 'r-', linewidth=2, zorder=11)
        ax.plot(est_pose[0], est_pose[1], 'ro', markersize=5, label='Estimated Pose')

        ax.legend(fontsize=15)
        ax.grid(True)
        ax.set_title(f"FastSLAM 2.0 - Real-time Visualization\nM = {M}, SENSOR_RANGE = {SENSOR_RANGE}, With noise", fontsize=15, loc='center')
        ax.axis('equal')
        ax.tick_params(axis='both', labelsize=14)
        ax.set_xlim(-30, 30)
        ax.set_ylim(-30, 30)
        plt.pause(0.01)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
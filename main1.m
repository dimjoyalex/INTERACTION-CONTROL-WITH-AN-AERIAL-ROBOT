function main()
    % Define initial position and desired position
    x0 = 0.0; y0 = 0.0; z0 = 0.280;     % Initial position
    x_desired = 2.000000; y_desired = 2.000000; z_desired = 0.3;   % Desired position

    % Design parameters
    b = 0.6;   % the length of total square cover by whole body of quadcopter in meters
    % Time duration for the trajectory 
    t_final = 30; % Example duration in seconds
    dt = 0.05;       % Time step
    speed_factor = 1.5;  % Speed factor
    m=1;
    g=9.81;
    [x_t, y_t, z_t, roll, pitch, yaw] = generate_trajectory(x0, y0, z0, x_desired, y_desired, z_desired, t_final, dt, speed_factor);
    x=x_t;
    y=y_t;
    z=z_t;
    fprintf('x = %.3f\n y = %.3f\n, z = %.3f', x_t,y_t,z_t); 
    params.alpha = 1; params.beta = 1; params.gamma = 1;
    params.kp = [0.5, 0.5, 0.5, 0.1]; % Position and force gains
    params.ki = [0.1, 0.1, 0.1, 0.05];
    params.kd = [0.5, 0.5, 0.5, 0.2];
    params.rho = [1, 1, 1, 0.5]; % Performance decay rates
    params.W_f = 0.5; % Example modeling error bound
    params.M_f = 0;               % No overshoot
    params.l_f = 4;               % Decay rate for performance bound
    params.rho_f0 = 2;            % Initial performance bound
    params.rho_f_inf = 0.01;      % Steady-state performance bound
    params.zeta_f = 0.01;      % Parameter from equation (23)
    params.delta = 0.1;        % Parameter from equation (24)
    params.sigma = 0.001;       % Regularization parameter
    params.kp_f = 0.5;                    % Proportional gain
    params.kd_f = 0.1;                    % Derivative gain
    params.ki_f = 0.01;                   % Integral gain
    params.nn_gain = 0.1;       % Neural network contribution gain
    params.damping_factor = 0.01; % Damping factor for height control
    params.learning_rate = 0.001; % Neural network learning rate
    params.regularization_factor = 0.005;  % Regularization constant
    params.ki_f = 0.001; % Integral gain
    params.velocity_gain = 0.02;    % Velocity feedback gain
    params.theta_rate = 0.001; % Learning rate for neural weights
    %params.gain_factor = 10;              % Gain factor for error transformation
    theta_hat_pos = zeros(6, 1);   % For position control
    theta_hat_force = zeros(2, 1); % For force control
    params.gain_decay_rate = 0.01;  % Decay rate for proportional gain
    params.u_base = m*g;        % Baseline thrust command (adjust as needed)
    % Initialize arrays for noisy measurements and Kalman estimates
    noisy_measurements = zeros(3, length(x)); % For storing noisy [x; y; z] measurements
    kalman_estimates = zeros(3, length(x)-1); % For storing Kalman filter estimates
    z_thrust_arr = zeros(length(x), 1);
    deformation_history = zeros(1, length(x)); % Initialize array to store deformation history
    position_error = zeros(1, length(x)); % Initialize array to store position errors
    % Initialize force history
    force_history = zeros(1, length(x));  % Initialize force history
    
    drone_height_history = zeros(1, length(x)); % Initialize array to store drone heights
    
    % Initialize the surface
    [X_surface, Y_surface, Z_surface] = initialize_surface();
    % Initialize deformation rate history
   
    % Animation and control loop
    i = 1; % Initialize the index
        % Initial surface deformation
    
    while i <= length(x)

       % Current desired position 
        x_target = x(i); y_target = y(i); 
        
        % Stop the simulation if within the threshold distance
        
        fprintf('\n\nTime: %.2f s',(i) * dt);
        
        % Generate noisy measurements
        noise_std = 0.02; % Measurement noise standard deviation
        noisy_measurement = [x(i); y(i); z(i)] + noise_std * randn(3, 1);
        noisy_measurements(:, i) = noisy_measurement; % Store the noisy measurement

        % Simple Kalman filter update
        if i == 1
            kalman_estimate = [x(i); y(i); z(i)];
            P = eye(3) * noise_std^2;
        else
            kalman_estimate = [x(i); y(i); z(i)];
            % Update the covariance matrix
            A = eye(3);
            Q = eye(3) * 0.001; % Process noise covariance
            P = A * P * A' + Q; 

            % Measurement update step
            H = eye(3);
            R = eye(3) * (noise_std)^2; 
            K = P * H' / (H * P * H' + R);
            kalman_estimate = kalman_estimate + K * (noisy_measurement - H * kalman_estimate);
            P = (eye(3) - K * H) * P;
        end
        kalman_estimates(:, i) = kalman_estimate; % Store the Kalman estimate
        % Print current Kalman filter estimate to the command window
        fprintf('\nKalman Estimate: x = %.4f, y = %.4f, z = %.4f\n',  kalman_estimates(1, i), kalman_estimates(2, i), kalman_estimates(3, i));
        def_ideal = (3/1000)^(2/3);
        fprintf('\nIdeal Height=%4f',0.3-def_ideal);
        [u_pos, theta_hat_pos,e_z] = position_neuro_adaptive_controller(x_target, y_target,0.3-def_ideal, ...
                                           kalman_estimates(1, i),kalman_estimates(2, i),kalman_estimates(3, i), theta_hat_pos, dt, params);
        u_pos=[u_pos(1);u_pos(2);u_pos(3);0];
        
        fprintf('NN Weights: %.7f %.7f %.7f %.7f %.7f %.7f\n', theta_hat_pos);
        [x_pred, y_pred, z_pred, roll_pred, pitch_pred, yaw_pred] = predict_next_state(kalman_estimates(1, i),kalman_estimates(2, i),kalman_estimates(3, i),u_pos, dt,e_z);
        
        [force, deformation] = calculate_force(x_pred, y_pred, 0.3-z_pred, X_surface, Y_surface, Z_surface);
        fprintf('force=%4f', force);  
        
          
        %Determine control mode based on contact
        if abs(force-3) < 0.05
            % No contact - use position control only
            fprintf('Using position control only - establishing contact\n');
            x(i)=x_pred;
            y(i)= y_pred;
            z(i)=z_pred; 
            roll(i)=roll_pred;
            pitch(i)=pitch_pred;
            yaw(i) = yaw_pred;
        else
            % Contact established - use force controller
            [u_f, theta_hat_force, e_f] = force_neuro_adaptive_controller(force, deformation, 3.0, ...
                                                                       theta_hat_force, dt, params, u_pos);
            fprintf('Force NN Output: %.7f %.7f', theta_hat_force); 
            % Format control input vector
            u_pos = [u_f; u_pos(2); u_pos(3); 0];
            [x(i), y(i), z(i), roll(i), pitch(i), yaw(i)] = simulate_drone_dynamics(x_pred,y_pred,z_pred, u_pos, dt,e_f); 
       end
       
       
        
       
        fprintf('\nroll = %.4f, pitch = %.4f, yaw = %.4f', roll(i), pitch(i), yaw(i));
        fprintf('\nFinal Position: x = %.4f, y = %.4f, z = %.4f\n',  x(i), y(i), z(i));
        
        % Store the height of the drone
        drone_height_history(i) = z(i); % Store the updated drone height

       
        %Calculate position error
        position_error(i) = sqrt((x(i) - x_target)^2 + (y(i) - y_target)^2);

        % Call the animation function to visualize the drone's position
        [force_sensor_reading, deformation] = drone_Animation(x(i), y(i), z(i), roll(i), pitch(i), yaw(i), x_desired, y_desired, 0.3-def_ideal, b, z(i), kalman_estimates, X_surface, Y_surface, Z_surface, i); 
        % Store deformation in the history array
        deformation_history(i) = deformation;
        force_history(i) = force_sensor_reading;
        pause(dt);  % Pause for the time step duration
        z_thrust_arr(i) = z(i);
        i = i + 1; % Increment the index
    end
    % Plot position error over time
    figure;
    plot(position_error);
    xlabel('Time Step');
    ylabel('Position Error (m)');
    title('Position Error Over Time');
    fprintf('lastx:%4f', x_t(i-1));
    % Plotting and visualization functions (existing code unchanged)
    plot_kalman_results(x, y, z, noisy_measurements, kalman_estimates);
    plot_force(force_history); % Plot force sensor reading
    plot_deformation(deformation_history);
end


function surface_z = get_surface_height(X_surface, Y_surface, Z_surface, x, y)
    % Interpolating the surface height using the input grid and points
    surface_z = interp2(X_surface, Y_surface, Z_surface, x, y, 'linear', NaN);
end

function [X_surface, Y_surface, Z_surface] = initialize_surface()
    [X_surface, Y_surface] = meshgrid(linspace(-5, 5, 25));
    Z_surface = zeros(size(X_surface)); % Flat surface initially
end

function [force_sensor_reading, deformation, Z_surface_updated] = drone_Animation(x, y, z, roll, pitch, yaw, x_desired, y_desired, ~, b, z_thrust, ~, X_surface, Y_surface, Z_surface, ~)
    % Define design parameters
    H = 0.06;  % height of drone in Z direction (4cm)
    H_m = H + H / 2; % height of motor in z direction (5 cm)
    r_p = b / 4;   % radius of propeller

    % Define Figure plot
    view(68, 53);
    grid on;
    axis equal;
    xlim([-5 5]);
    ylim([-5 5]);
    zlim([-5 5]);
    title('Drone Animation')
    xlabel('X [m]');
    ylabel('Y [m]');
    zlabel('Z [m]');
    hold on;

    % Plot the desired trajectory with adjusted height
    plot3([x(1), x_desired], [y(1), y_desired], [z(1)-0.3, 0], 'k--', 'LineWidth', 2);
    
    % Plot the planar blue surface
    surface(X_surface, Y_surface, Z_surface, 'FaceColor', 'blue', 'FaceAlpha', 0.3);
    
    % Clear previous plot
    cla;
     % Plot the desired trajectory with adjusted height
    plot3([x(1), x_desired], [y(1), y_desired], [z(1)-0.3, 0], 'k--', 'LineWidth', 2);
   
    % Plot the drone at the estimated position
    plot_drone(x, y, z, roll, pitch, yaw, b, H, H_m, r_p);

    % Plot the probe at the desired position
    [force_sensor_reading, deformation] = plot_probe(x, y, z_thrust, X_surface, Y_surface, Z_surface);

    % Update the surface by adjusting Z_surface according to the deformation
    Z_surface_updated = Z_surface;  % Start with original surface heights
    deformation_area_radius = 0.3; % Radius around probe where deformation is applied (adjust as necessary)
    
    % Apply deformation to surface based on probe location
    for j = 1:length(X_surface)
        for k = 1:length(Y_surface)
            dist_to_probe = sqrt((X_surface(j, k) - x)^2 + (Y_surface(j, k) - y)^2);
            if dist_to_probe <= deformation_area_radius
                % Apply deformation to Z_surface (sink the surface)
                Z_surface_updated(j, k) = Z_surface(j, k) - deformation*20;% *20 so that we can visually see te deformation of the surface
            end
        end
    end

    % Plot the blue surface with updated Z_surface
    surface(X_surface, Y_surface, Z_surface_updated, 'FaceColor', 'blue', 'FaceAlpha', 0.3); 

    % Set the current frame
    drawnow; 
    hold off;
end


function [x, y, z, roll, pitch, yaw] = generate_trajectory(x0, y0, z0, x_desired, y_desired, z_desired, t_final, dt, speed_factor)
    % Generate a straight-line trajectory from initial position to desired position
    t = 0:dt:t_final; 
    % Adjust the number of steps based on speed_factor
    num_steps = ceil(numel(t) / speed_factor);
    x = linspace(x0, x_desired, num_steps);
    y = linspace(y0, y_desired, num_steps);
    z = linspace(z0, z_desired, num_steps);
    
    % Assume constant roll, pitch, and yaw for simplicity
    roll = zeros(size(x));     % No roll
    pitch = zeros(size(x));    % No pitch
    yaw = zeros(size(x));      % No yaw
end

function [x_new, y_new, z_new, roll_new, pitch_new, yaw_new] = simulate_drone_dynamics(x, y, z, u, dt,e_f)
    %Physical parameters
    m = 1;  % Mass of the quadcopter (kg)
    
    Ixx = 0.1; % Moment of inertia about x-axis
    Iyy = 0.1; % Moment of inertia about y-axis
    Izz = 0.3; % Moment of inertia about z-axis
    L = 0.5;  % Distance from the center of the quadcopter to any of the propellers (m)
    k=0.1;
    b = 0.01; % Torque constant for yaw

    % Extract control inputs (Throttle, Roll, Pitch, Yaw)
    throttle = u(1);  % Thrust control input
    roll_in = u(2);   % Roll input
    pitch_in = u(3);  % Pitch input
    yaw_in = u(4);    % Yaw input
    
    % Motor thrusts (no change in logic)
    T1 = throttle + roll_in + pitch_in + yaw_in;
    T2 = throttle - roll_in + pitch_in - yaw_in;
    T3 = throttle + roll_in - pitch_in - yaw_in;
    T4 = throttle - roll_in - pitch_in + yaw_in;
    
    % Motor angular velocities

    omega1=sqrt(T1);
    omega2=sqrt(T2);
    omega3=sqrt(T3);
    omega4=sqrt(T4);
    
    
    k1=abs(0.1*e_f);
    total_thrust = k1*(omega1^2 + omega2^2 + omega3^2 + omega4^2);
    
    % Compute torques for roll, pitch, and yaw (unchanged logic)
    tau_phi = L * k * (-omega2^2 + omega4^2);  % Roll torque
    tau_theta = L * k * (-omega1^2 + omega3^2);  % Pitch torque
    tau_psi = b * (omega1^2 - omega2^2 + omega3^2 - omega4^2);  % Yaw torque

    % Update linear accelerations (using Newton-Euler equations)
    
    z_acceleration = -sign(e_f)*(total_thrust / m);  % Vertical acceleration (gravity + thrust)
   
    x_acceleration = total_thrust / m * (cos(yaw_in) * sin(pitch_in) * cos(roll_in) + sin(yaw_in) * sin(roll_in));
    y_acceleration = total_thrust / m * (sin(yaw_in) * sin(pitch_in) * cos(roll_in) - cos(yaw_in) * sin(roll_in));

    % Update angular accelerations (using Euler-Lagrange equations)
    p = roll_in;
    q = pitch_in;
    r = yaw_in;

    % Compute angular accelerations based on the torques
    roll_acceleration = (tau_phi - Izz * q * r) / Ixx;
    pitch_acceleration = (tau_theta - Ixx * p * r) / Iyy;
    yaw_acceleration = (tau_psi - Iyy * p * q) / Izz;

    % Calculate velocities first (proper Euler integration)
    vx = x_acceleration * dt;
    vy = y_acceleration * dt;
    vz = z_acceleration * dt; 

    % Predict next state using Euler integration (single dt multiplication)
    x_new = x + vx * dt;
    y_new = y + vy * dt;
    z_new = z + vz * dt;

    roll_new = roll_in + roll_acceleration * dt;
    pitch_new = pitch_in + pitch_acceleration * dt;
    yaw_new = yaw_in + yaw_acceleration * dt;

    % Debugging information
    fprintf('\nThrottle Input: %.4f', throttle);
    fprintf(' Total Thrust: %.4f N', total_thrust);
    fprintf(' Z Acceleration: %.4f m/s^2', z_acceleration);
    
end


function [x_pred, y_pred, z_pred, roll_pred, pitch_pred, yaw_pred] = predict_next_state(x, y, z, u, dt,e_z)
   %%Physical parameters
    m = 1;  % Mass of the quadcopter (kg)
    
    Ixx = 0.1; % Moment of inertia about x-axis
    Iyy = 0.1; % Moment of inertia about y-axis
    Izz = 0.3; % Moment of inertia about z-axis
    L = 0.5;  % Distance from the center of the quadcopter to any of the propellers (m)
    k=0.1;
 
    b = 0.01; % Torque constant for yaw

    % Extract control inputs (Throttle, Roll, Pitch, Yaw)
    throttle = u(1);  % Thrust control input
    roll_in = u(2);   % Roll input
    pitch_in = u(3);  % Pitch input
    yaw_in = u(4);    % Yaw input
    
    % Motor thrusts (no change in logic)
    T1 = throttle + roll_in + pitch_in + yaw_in;
    T2 = throttle - roll_in + pitch_in - yaw_in;
    T3 = throttle + roll_in - pitch_in - yaw_in;
    T4 = throttle - roll_in - pitch_in + yaw_in;
    
    % Motor angular velocities

    omega1=sqrt(T1);
    omega2=sqrt(T2);
    omega3=sqrt(T3);
    omega4=sqrt(T4);
    
    
    k1=10*abs(e_z);
    total_thrust = k1*(omega1^2 + omega2^2 + omega3^2 + omega4^2);

    % Compute torques for roll, pitch, and yaw (unchanged logic)
    tau_phi = L * k * (-omega2^2 + omega4^2);  % Roll torque
    tau_theta = L * k * (-omega1^2 + omega3^2);  % Pitch torque
    tau_psi = b * (omega1^2 - omega2^2 + omega3^2 - omega4^2);  % Yaw torque

    % Update linear accelerations (using Newton-Euler equations)
    
    z_acceleration = sign(e_z)*(total_thrust / m);  % Vertical acceleration (gravity + thrust)
   
    x_acceleration = total_thrust / m * (cos(yaw_in) * sin(pitch_in) * cos(roll_in) + sin(yaw_in) * sin(roll_in));
    y_acceleration = total_thrust / m * (sin(yaw_in) * sin(pitch_in) * cos(roll_in) - cos(yaw_in) * sin(roll_in));

    % Update angular accelerations (using Euler-Lagrange equations)
    p = roll_in;
    q = pitch_in;
    r = yaw_in;

    % Compute angular accelerations based on the torques
    roll_acceleration = (tau_phi - Izz * q * r) / Ixx;
    pitch_acceleration = (tau_theta - Ixx * p * r) / Iyy;
    yaw_acceleration = (tau_psi - Iyy * p * q) / Izz;

    % Calculate velocities first (proper Euler integration)
    vx = x_acceleration * dt;
    vy = y_acceleration * dt;
    vz = z_acceleration * dt;
    % Predict next state using Euler integration (single dt multiplication)
    x_pred = x + vx * dt;
    y_pred= y + vy * dt;
    z_pred= z + vz * dt;

    roll_pred = roll_in + roll_acceleration * dt;
    pitch_pred = pitch_in + pitch_acceleration * dt;
    yaw_pred = yaw_in + yaw_acceleration * dt;
    
    % Add debug output to verify predictions
    
    fprintf('Predicted State:\n');
    fprintf('Position (x,y,z): %.4f, %.4f, %.4f\n', x_pred, y_pred, z_pred);
    fprintf('\nThrottle Input: %.4f', throttle);
    fprintf(' Z acceleration: %.4f, Z velocity: %.4f\n', z_acceleration, vz);
    
end



function [u_pos, theta_hat_pos,e_z] = position_neuro_adaptive_controller(x_target, y_target, z_target, x, y, z, theta_hat_pos, dt, params)
    % Extract parameters
    m=1;
    g=9.81;
    alpha = params.alpha; beta = params.beta; gamma = params.gamma;
    kp = params.kp(1:3); ki = params.ki(1:3); kd = params.kd(1:3);
    rho = params.rho(1:3); % Performance function decay rates
    theta_rate = params.theta_rate; % Learning rate
    %fprintf('NN Weights: %.7f %.7f %.7f %.7f %.7f %.7f\n', theta_hat_pos);
    % Position errors
    e_pos = [x_target - x; y_target - y; z_target - z];
    
    % Initialize persistent variable for previous error
    persistent e_pos_prev;
    if isempty(e_pos_prev)
        e_pos_prev = e_pos;
    end
    e_z = z_target - z;
    % Prescribed performance error transformation
    epsilon_pos = e_pos ./ (rho + eps);

    % Derivative and integral of the error
    e_diff = (e_pos - e_pos_prev) / dt;
    lambda = 0.01;
    e_int = cumsum(e_pos * dt);
    e_int = e_int * (1 - lambda) + e_pos * dt;

    % Lyapunov-based terms
    s_pos = alpha * (kp .* epsilon_pos + kd .* e_diff + ki .* e_int);

    % Neural network approximation
    z_vector = [x; y; z; x_target; y_target; z_target];
    neural_output = gamma * (theta_hat_pos' * z_vector);

    % Adaptive law for neural network weights
    theta_hat_dot = beta * theta_rate * z_vector * norm(s_pos);
    theta_hat_pos = theta_hat_pos + theta_hat_dot * dt;

    % Compute control input
    u_pos = s_pos + neural_output;
    u_pos(1)=u_pos(1) + m*g;
    % Update previous error
    e_pos_prev = e_pos;
end



function [u_force, theta_hat_force, e_f] = force_neuro_adaptive_controller(force_current, deformation, force_desired, theta_hat_force, dt, params,u_pos)
    % Implements a stable neuro-adaptive force controller based on the pape
    gamma = 0.01;  % Reduced gamma value for stability
    rho_f0 = params.rho_f0;
    rho_f_inf = params.rho_f_inf;
    l_f = params.l_f;
    learning_rate = 0.9;  % Reduced learning rate for stability
    sigma = params.sigma;
    
    % Persistent variables for state tracking
    persistent rho_f e_f_prev e_f_int prev_u_force;
    if isempty(rho_f)
        rho_f = rho_f0;
        e_f_prev = 0;
        e_f_int = 0;
        prev_u_force = 0;
    end
    
    % Update performance function with numerical safety
    rho_f_dot = -l_f * (rho_f - rho_f_inf);
    rho_f = max(rho_f + rho_f_dot * dt, 0.01); % Ensure positive rho_f
    
    % Force error and its derivative (with filtering)
    e_f = force_desired - force_current;
    e_f_dot = (e_f - e_f_prev) / dt;
    
    % Apply low-pass filter to error derivative
    filter_coef = 0.7;
    e_f_dot = filter_coef * e_f_dot + (1-filter_coef) * e_f_prev / dt;
    e_f_prev = e_f;
    
    % Integrate error (with anti-windup)
    lambda = 0.1;  % Increased forgetting factor
    max_integral = 0.2;  % Anti-windup limit
    e_f_int = e_f_int * (1 - lambda) + e_f * dt;
    e_f_int = max(min(e_f_int, max_integral), -max_integral);  % Apply anti-windup
    
    % Simplified control approach for stability
    % Use a PID-like structure with neural network adaptation
    kp = 0.01;
    ki = 0.01;
    kd = 0.05;
    
    % Neural network input vector
    Z_f = [deformation; 0.279199]; % Input: deformation and bias
    
    % Neural network output (approximation of partial_f)
    partial_f_hat = theta_hat_force' * Z_f; % Ensure positive value
    
    % Update neural network weights (simplified)
    adaptation_error = e_f;
    theta_hat_dot = learning_rate * (adaptation_error * Z_f - sigma * theta_hat_force);
    
    % Limit weight update magnitude
    max_update = 0.001;
    theta_hat_norm = norm(theta_hat_dot);
    if theta_hat_norm > max_update
        theta_hat_dot = theta_hat_dot * (max_update / theta_hat_norm);
    end
    
    theta_hat_force = theta_hat_force + theta_hat_dot * dt;
    
    % Calculate control input using PID + neural compensation
    u_pid = kp * e_f + ki * e_f_int + kd * e_f_dot;
    u_nn = gamma * partial_f_hat * sign(e_f);
    
    % Combine and limit control output
    u_force = u_pid + u_nn;
    
    % Rate limiter
    max_rate = 0.01;
    if abs(u_force - prev_u_force) > max_rate
        u_force = prev_u_force + max_rate * sign(u_force - prev_u_force);
    end
      
    % Store for next iteration
    prev_u_force = u_force;
    u_force=u_pos(1)+u_force;
    % Debug output
    fprintf(' Force Control: Current=%.3fN, Desired=%.3fN, Error=%.3fN\n', ...
            force_current, force_desired, e_f);
    
   
end


function [force_sensor_reading, deformation] = plot_probe(x, y, z, X_surface, Y_surface, Z_surface)
    probe_length = 0.3; % Adjust as necessary
    probe_radius = 0.01; % Adjust as necessary

    % Plot the probe cylinder
    [xcylinder_probe, ycylinder_probe, zcylinder_probe] = cylinder(probe_radius, 20);
    surface(xcylinder_probe + x, ycylinder_probe + y, -zcylinder_probe * probe_length + z, 'facecolor', 'k');
    
    % Calculate and apply force and deformation
    [force_sensor_reading, deformation] = calculate_force(x, y, 0.3 - z, X_surface, Y_surface, Z_surface);

    % Display force reading
   
    text(x + 0.1, y + 0.1, z + 0.8, ['      Force(N): ', num2str(force_sensor_reading, '%.4f')], ...
         'FontSize', 15, 'FontWeight', 'bold', 'Color', 'red');
    fprintf('Final Force = %4f', force_sensor_reading);
end


function plot_drone(x, y, z, roll, pitch, yaw, b, H, H_m, r_p)
    % Plot the drone at the given position and orientation
    % Define design parameters
    %D2R = pi / 180;
    %ro = 45 * D2R;                   % angle by which rotate the base of quadcopter
     % Precompute the cosines and sines of the roll, pitch, and yaw angles
    C_phi = cos(roll);
    S_phi = sin(roll);
    C_theta = cos(pitch);
    S_theta = sin(pitch);
    C_psi = cos(yaw);
    S_psi = sin(yaw);

    % Rotation matrix using the provided formula
    Ri = [
        C_psi * C_theta,                  S_psi * C_theta,                 -S_theta;
        C_psi * S_theta * S_phi - S_psi * C_phi, S_psi * S_theta * S_phi + C_psi * C_phi, C_theta * S_phi;
        C_psi * S_theta * C_phi + S_psi * S_phi, S_psi * S_theta * C_phi - C_psi * S_phi, C_theta * C_phi
    ];

    
    
    base_co = [-b / 6, b / 6, b / 6, -b / 6; % Coordinates of Base 
               -b / 6, -b / 6, b / 6, b / 6;
               0,    0,   0,   0];
    base = Ri * base_co + [x; y; z - H]; % rotate base Coordinates by 45 degree and adjust z for drone height

    to = linspace(0, 2 * pi);
    xp = r_p * cos(to) + x;
    yp = r_p * sin(to) + y;
    zp = zeros(1, length(to)) + z;

    % Plot drone components
    % Design the base square
    patch(base(1,:), base(2,:), base(3,:), 'r');
    patch(base(1,:), base(2,:), base(3,:) + H, 'r');

    % Design 2 perpendicular legs of quadcopter 
    [xcylinder, ycylinder, zcylinder] = cylinder([H / 2, H / 2]);
    surface(b * zcylinder - b / 2 + x, ycylinder + y, xcylinder + H / 2 + z, 'facecolor', 'b');
    surface(ycylinder + x, b * zcylinder - b / 2 + y, xcylinder + H / 2 + z, 'facecolor', 'b');

    % Design 4 cylindrical motors 
    surface(xcylinder + b / 2 + x, ycylinder + y, H_m * zcylinder + H / 2 + z, 'facecolor', 'r');
    surface(xcylinder - b / 2 + x, ycylinder + y, H_m * zcylinder + H / 2 + z, 'facecolor', 'r');
    surface(xcylinder + x, ycylinder + b / 2 + y, H_m * zcylinder + H / 2 + z, 'facecolor', 'r');
    surface(xcylinder + x, ycylinder - b / 2 + y, H_m * zcylinder + H / 2 + z, 'facecolor', 'r');

    % Design 4 propellers
    patch(xp + b / 2, yp, zp + (H_m + H / 2), 'c', 'LineWidth', 0.5);
    patch(xp - b / 2, yp, zp + (H_m + H / 2), 'c', 'LineWidth', 0.5);
    patch(xp, yp + b / 2, zp + (H_m + H / 2), 'p', 'LineWidth', 0.5);
    patch(xp, yp - b / 2, zp + (H_m + H / 2), 'p', 'LineWidth', 0.5);
end

function [force, deformation] = calculate_force(x_probe, y_probe, z_probe, X_surface, Y_surface, Z_surface)
    % Get the surface height under the probe position
    surface_z = get_surface_height(X_surface, Y_surface, Z_surface, x_probe, y_probe);
    deformation = (z_probe - surface_z); % How much the probe penetrates
    
    % Initialize force as zero
    force = 0;
   
    % Check for meaningful deformation
    if deformation > 0.0001  
        k = 1000; % Surface stiffness constant
        force = k * deformation^1.5; % Apply a nonlinear deformation force
        
    end
    
end

function plot_kalman_results(x, y, z, noisy_measurements, kalman_estimates)
    % This function plots the results of the Kalman filter alongside the actual trajectory
    % and the noisy measurements.
    
    figure; % Create a new figure
    hold on; grid on; % Enable grid and hold to plot multiple datasets
    
    % Plot actual trajectory
    plot3(x, y, z, 'g-', 'LineWidth', 2, 'DisplayName', 'Actual Trajectory');
    
    % Plot noisy measurements
    plot3(noisy_measurements(1, :), noisy_measurements(2, :), noisy_measurements(3, :), 'kx', 'DisplayName', 'Noisy Measurements');
    
    % Plot Kalman filter estimates
    plot3(kalman_estimates(1, :), kalman_estimates(2, :), kalman_estimates(3, :), 'b-', 'LineWidth', 2, 'DisplayName', 'Kalman Estimated Position');
    
    % Enhancements for visualization
    xlabel('X [m]');
    ylabel('Y [m]');
    zlabel('Z [m]');
    zlim([0, 0.4]);
    title('Drone Trajectory, Noisy Measurements, and Kalman Filter Estimates');
    legend show; % Display legend to identify each dataset
    % Set the view angle
    view(3); % Default 3D view
end

function plot_force(force_history)
    % Plot force sensor reading over time in a separate figure
    figure, 
    plot(1:length(force_history), force_history, 'r', 'LineWidth', 1.5);
    xlabel('Time Step');
    ylabel('Force (N)');
    title('Force Sensor Reading over Time');
    grid on;
    %ylim([-2, 0]);
end

function plot_deformation(deformation_history)
    % Plot the deformation over time
    figure;
    plot(1:length(deformation_history), deformation_history, 'r-', 'LineWidth', 2);
    xlabel('Time step');
    ylabel('Deformation (m)');
    title('Deformation over Time');
    grid on;
end
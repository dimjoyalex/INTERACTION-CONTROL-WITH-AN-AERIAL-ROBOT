function main()
    % Define initial position and desired position
    x0 = 0.0; y0 = 0.0; z0 = 0.30;     % Initial position
    x_desired = 2.000000; y_desired = 2.000000; z_desired = 0.30;   % Desired position

    % Design parameters
    b = 0.6;   % the length of total square cover by whole body of quadcopter in meters
    % Time duration for the trajectory 
    t_final = 30; % Example duration in seconds
    dt = 0.1;       % Time step
    speed_factor = 1.5;  % Speed factor

    [x_t, y_t, z_t, roll, pitch, yaw] = generate_trajectory(x0, y0, z0, x_desired, y_desired, z_desired, t_final, dt, speed_factor);
    x=x_t;
    y=y_t;
    z=z_t;
    %fprintf('x = %.3f\n y = %.3f\n, z = %.3f', x_t,y_t,z_t); 
    params.alpha = 1; params.beta = 1; params.gamma = 1;
    params.kp = [0.5, 0.5, 1, 0.1]; % Position and force gains
    params.ki = [0.1, 0.1, 0.1, 0.05];
    params.kd = [0.5, 0.5, 0.5, 0.2];
    params.rho = [1, 1, 1, 0.5]; % Performance decay rates
    params.W_f = 0.5; % Example modeling error bound
    params.theta_rate = 0.9; % Learning rate for neural weights
    theta_hat = zeros(6, 1); % Initialize NN weights


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
    
    % Animation and control loop
    i = 1; % Initialize the index
    while i <= length(x)

        % Current desired position 
        x_target = x_t(i); y_target = y_t(i); z_target = z_t(i);

        
        fprintf('\n\nTime: %.2f s',(i-1) * dt);
        
        % Generate noisy measurements
        noise_std = 0.01; % Measurement noise standard deviation
        noisy_measurement = [x(i); y(i); z(i)] + noise_std * randn(3, 1);
        noisy_measurements(:, i) = noisy_measurement; % Store the noisy measurement

        % Improved Kalman filter update
        if i == 1
            kalman_estimate = [0;0;0.3];
            P = eye(3) * noise_std^2;
        else
            % State prediction incorporating control inputs
            A = eye(3); % State transition model
            Q = eye(3) * 0.001; % Process noise covariance
            control_input = get_control_input(u_pos, dt);

            kalman_estimate = A * kalman_estimate + control_input; 
            P = A * P * A' + Q; % Predict error covariance

            H = eye(3);
            R = eye(3) * (noise_std)^2; % Measurement noise covariance
            K = P * H' / (H * P * H' + R); % Kalman gain
            kalman_estimate = kalman_estimate + K * (noisy_measurement - H * kalman_estimate); % Update estimate
            P = (eye(3) - K * H) * P; % Update error covariance
        end
        
        kalman_estimates(:, i) = kalman_estimate; % Store the Kalman estimate
        % Print current Kalman filter estimate to the command window
        fprintf('\nKalman Estimate: x = %.4f, y = %.4f, z = %.4f\n',  kalman_estimates(1, i), kalman_estimates(2, i), kalman_estimates(3, i));
        [u_pos, theta_hat] = neuro_adaptive_controller(x_target, y_target, z_target, ...
                                           kalman_estimates(1, i),kalman_estimates(2, i),kalman_estimates(3, i), theta_hat, dt, params);
        
        % Combine control outputs (consider only vertical thrust from altitude controller)
        u = [u_pos(1); u_pos(2); u_pos(3); 0];  % Assuming no yaw control for simplicity
        u = altitude_limiter(kalman_estimates(3, i), u);
        % Simulate drone response 
        [x(i), y(i), z(i), roll(i), pitch(i), yaw(i)] = simulate_drone_dynamics(x(i), y(i), z(i), u, dt);
        
        % Limit the altitude using the altitude_limiter function
        fprintf('Position: x = %.4f, y = %.4f, z = %.4f\n',  x(i), y(i), z(i));
        
        u = altitude_limiter(z(i), u);  % Adjust the vertical control if necessary
        
        % Simulate drone response 
        [x(i), y(i), z(i), roll(i), pitch(i), yaw(i)] = simulate_drone_dynamics(x(i), y(i), z(i), u, dt);
        %fprintf('u = %.4f  ', u);
        fprintf('roll = %.4f, pitch = %.4f, yaw = %.4f', roll(i), pitch(i), yaw(i));
        fprintf('\nFinal Position: x = %.4f, y = %.4f, z = %.4f\n',  x(i), y(i), z(i));
        
        % Store the height of the drone
        drone_height_history(i) = z(i); % Store the updated drone height
        fprintf('NN Weights: %.7f %.7f %.7f %.7f %.7f %.7f\n', theta_hat);
        % Print the height after applying thrust
        fprintf('Height: %.4f\n', z(i));
        %Calculate position error
        position_error(i) = sqrt((x(i) - x_target)^2 + (y(i) - y_target)^2);

        % Call the animation function to visualize the drone's position
        [force_sensor_reading, deformation] = drone_Animation(x, y, z, roll, pitch, yaw, x_desired, y_desired, z_desired, b, z(i), kalman_estimates, X_surface, Y_surface, Z_surface, i); 
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
    % Plotting and visualization functions (existing code unchanged)
    plot_kalman_results(x, y, z, noisy_measurements, kalman_estimates);
    plot_force(force_history); % Plot force sensor reading
    plot_deformation(deformation_history);
end

function control_input = get_control_input(u_pos, dt)
    control_input = [0; 0; u_pos(3) * dt]; % Adjust the input for vertical thrust
end


function surface_z = get_surface_height(X_surface, Y_surface, Z_surface, x, y)
    % Interpolating the surface height using the input grid and points
    surface_z = interp2(X_surface, Y_surface, Z_surface, x, y, 'linear', NaN);
end

function [X_surface, Y_surface, Z_surface] = initialize_surface()
    [X_surface, Y_surface] = meshgrid(linspace(-5, 5, 25));
    Z_surface = zeros(size(X_surface)); % Flat surface initially
end

function [force_sensor_reading, deformation, Z_surface_updated] = drone_Animation(x, y, z, roll, pitch, yaw, x_desired, y_desired, z_desired, b, z_thrust, kalman_estimates, X_surface, Y_surface, Z_surface, i)
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
    plot3([x(1), x_desired], [y(1), y_desired], [z(1)-0.3, z_desired-0.3], 'k--', 'LineWidth', 2);
    
    % Plot the planar blue surface
    %surface(X_surface, Y_surface, Z_surface, 'FaceColor', 'blue', 'FaceAlpha', 0.3);
    
    % Clear previous plot
    cla;
    
    % Plot the trajectory with adjusted height
    plot3([x(1), x_desired], [y(1), y_desired], [z(1)-0.3, z_desired-0.3], 'k--', 'LineWidth', 2);
   
    % Plot the drone at the estimated position
    plot_drone(x(i), y(i), z(i), roll(i), pitch(i), yaw(i), b, H, H_m, r_p);

    % Plot the probe at the desired position
    [force_sensor_reading, deformation] = plot_probe(x(i), y(i), z_thrust, X_surface, Y_surface, Z_surface);

    % Update the surface by adjusting Z_surface according to the deformation
    Z_surface_updated = Z_surface;  % Start with original surface heights
    deformation_area_radius = 0.3; % Radius around probe where deformation is applied (adjust as necessary)
    
    % Apply deformation to surface based on probe location
    for j = 1:length(X_surface)
        for k = 1:length(Y_surface)
            dist_to_probe = sqrt((X_surface(j, k) - x(i))^2 + (Y_surface(j, k) - y(i))^2);
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


function [x_new, y_new, z_new, roll_new, pitch_new, yaw_new] = simulate_drone_dynamics(x, y, z, u, dt)
    % Physical parameters
    m = 1;  % mass of the quadcopter (kg)
    g = 9.81;  % acceleration due to gravity (m/s^2)
    Ixx = 0.1; % Moment of inertia about x-axis
    Iyy = 0.1; % Moment of inertia about y-axis
    Izz = 0.3; % Moment of inertia about z-axis
    L = 0.5;  % distance from the center of the quadcopter to any of the propellers (m)
      % some appropriately dimensioned constant for drag torque
    
    % Extract control inputs (Throttle, Roll, Pitch, Yaw)
    throttle = u(1);  % Thrust control input
    roll_in = u(2);   % Roll input
    pitch_in = u(3);  % Pitch input
    yaw_in = u(4);    % Yaw input

    % Assume omega (angular velocities for each motor)
    % These can be derived from your control algorithm
    omega1 = throttle + roll_in + pitch_in + yaw_in;  % Placeholder values
    omega2 = throttle - roll_in + pitch_in - yaw_in;  % Placeholder values
    omega3 = throttle + roll_in - pitch_in - yaw_in;  % Placeholder values
    omega4 = throttle - roll_in - pitch_in + yaw_in;  % Placeholder values
    
    % Constants
    k = 0.1;  % Thrust constant 
    kl = k*L;  
    b = 0.01;  % Torque constant for yaw 
    
    % Compute total thrust (T) from angular velocities
    total_thrust = k * (omega1^2 + omega2^2 + omega3^2 + omega4^2);
    
    % Compute torques for roll, pitch, and yaw
    tau_phi = kl * (-omega2^2 + omega4^2);  % Roll torque
    tau_theta = kl * (-omega1^2 + omega3^2);  % Pitch torque
    tau_psi = b * (omega1^2 - omega2^2 + omega3^2 - omega4^2);  % Yaw torque
    
    % Update linear accelerations (using Newton-Euler equations)
    z_acceleration = -g + total_thrust / m;  % Vertical acceleration (gravity + thrust)
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

    % Update positions and angles using Euler integration
    x_new = x + dt * cos(pitch_in) * cos(yaw_in) * cos(roll_in) * dt * x_acceleration;
    y_new = y + dt * (sin(roll_in) * sin(pitch_in) * cos(yaw_in) - cos(roll_in) * sin(yaw_in)) * dt * y_acceleration;
    z_new = z + dt * (cos(roll_in) * sin(pitch_in) * cos(yaw_in) + sin(roll_in) * sin(yaw_in)) * dt * z_acceleration;
    roll_new = roll_in + dt * roll_acceleration;
    pitch_new = pitch_in + dt * pitch_acceleration;
    yaw_new = yaw_in + dt * yaw_acceleration;
end


function [u, theta_hat] = neuro_adaptive_controller(x_target, y_target, z_target, x, y, z, theta_hat, dt, params)
    % Extract parameters
    
    alpha = params.alpha; beta = params.beta; gamma = params.gamma;
    kp = params.kp; ki = params.ki; kd = params.kd;
    rho = params.rho; % Performance function decay rates
    theta_rate = params.theta_rate; % Learning rate

    % Position and force errors
    e_pos = [x_target - x; y_target - y; 0.3 - z];
    e_force = 0.3 - z;
    
    % Initialize persistent variable for previous error
    persistent e_pos_prev;
    if isempty(e_pos_prev)
        e_pos_prev = e_pos; % Initialize with the first error
    end

    % Prescribed performance error transformation
    epsilon_pos = e_pos ./ (rho(1:3) + eps); % Avoid division by zero
    epsilon_force = e_force / (rho(4) + eps); 
    % Derivative and integral of the error
    e_diff = (e_pos - e_pos_prev) / dt; % Numerical derivative
    lambda = 0.01; % Small decay rate to avoid excessive accumulation
    
    e_int = cumsum(e_pos * dt);         % Integral over time (should reset each call)
    e_int = e_int * (1 - lambda) + e_pos * dt;
    % Lyapunov-based terms with alpha and beta scaling
    s_pos = alpha * (kp(1:3) .* epsilon_pos + kd(1:3) .* e_diff + ki(1:3) .* e_int);
    
    s_force = alpha * (kp(4) * epsilon_force); % For force control (simplified here)

    % Neural network approximation for unknown dynamics with gamma scaling
    z_vector = [x; y; z; x_target; y_target; z_target]; % Example input
    neural_output = gamma * (theta_hat' * z_vector);    % Scaled neural output

    % Adaptive law for neural network weights with beta scaling
    theta_hat_dot = beta * theta_rate * z_vector * s_force;  % Update rule
    theta_hat = theta_hat + theta_hat_dot * dt;

    % Compute control input
    u = s_pos + neural_output; % Combine feedback and feedforward terms

    % Update previous error
    e_pos_prev = e_pos;

    % Thrust adjustment based on force control (Ensuring force within bounds)
    % Example: Adjust thrust to keep force within 1-5 N range
    force_threshold_min = 1; % Minimum force (1 N)
    force_threshold_max = 3; % Maximum force (5 N)

    if s_force < force_threshold_min
        u(3) = u(3) + 0.8*(force_threshold_min - s_force); % Increase thrust if force is too low
    elseif s_force > force_threshold_max
        u(3) = u(3) - (s_force - force_threshold_max); % Decrease thrust if force is too high
    end
end


function u = altitude_limiter(z, u)
    z_threshold = 0.3;  % Maximum allowable height

    % Check if the drone exceeds the height threshold
    if z >= z_threshold
        % If the drone is at or above the threshold and the control is pushing upwards
        u(3) = abs((1.5*u(3)));  % Stop any upward thrust
    end
    if z <= 0.297
        u(3) = -abs((0.7*u(3)));  
    end
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
   
    text(x + 0.1, y + 0.1, z + 0.2, ['      Force(N): ', num2str(force_sensor_reading, '%.4f')], ...
         'FontSize', 15, 'FontWeight', 'bold', 'Color', 'red');

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

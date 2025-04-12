function main()
    % Define initial position and desired position
    x0 = 0.0; y0 = 0.0; z0 = 0.280;     % Initial position
    x_desired = 2; y_desired = 2; z_desired = 0.280;   % Desired position

    % Design parameters
    b = 0.6;   % the length of total square cover by whole body of quadcopter in meters
    t_final = 20; % Example duration in seconds
    dt = 0.1;       % Time step
    speed_factor = 1.5;  % Speed factor

    [x, y, z, roll, pitch, yaw] = generate_trajectory(x0, y0, z0, x_desired, y_desired, z_desired, t_final, dt, speed_factor);
    
    % Initialize PID controller parameters 
    kp_pos = [0.5, 0.5, 1]; % Proportional gains for x, y, z
    ki_pos = [0.05, 0.05, 0.05]; % Integral gains for x, y, z
    kd_pos = [0.1, 0.1, 0.1]; % Derivative gains for x, y, z
    
    % Initial PID errors
    e_prev_pos = [0; 0; 0]; % Previous error for position
    e_int_pos = [0; 0; 0];  % Integral of position error
    position_error = zeros(1, length(x));
    % Initialize arrays for noisy measurements and Kalman estimates
    noisy_measurements = zeros(3, length(x)-1); % For storing noisy [x; y; z] measurements
    kalman_estimates = zeros(3, length(x)-1); % For storing Kalman filter estimates
    z_thrust_arr = zeros(length(x), 1);
    deformation_history = zeros(1, length(x)-1); % Initialize array to store deformation history

    % Initialize force history
    force_history = zeros(1, length(x)-1);  % Initialize force history
    
    % Initialize the surface
    [X_surface, Y_surface, Z_surface] = initialize_surface();
    drone_height_history = zeros(1, length(x)); % Initialize array to store drone heights

    % Define convergence threshold
    threshold_distance = 0.01; % Threshold distance (meters)

    % Animation and control loop
    i = 1; % Initialize the index
    while i <= length(x)-1
        
        % Current desired position 
        x_target = x(i); y_target = y(i); z_target = z(i);

        % Calculate the distance to the desired position
        distance_to_target = sqrt((x_target - x_desired)^2 + (y_target - y_desired)^2 + (z_target - z_desired)^2);
        
        % Stop the simulation if within the threshold distance
        if distance_to_target < threshold_distance
            break; % Exit the loop if close enough to the target
        end
        fprintf('\n\nTime: %.2f s',(i) * dt);
        
        % Generate noisy measurements
        noise_std = 0.05; % Measurement noise standard deviation
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
        % Call the position PID controller
        [u_pos, e_prev_pos, e_int_pos] = pid_controller(x_target, y_target, 0.28, ...
                                    kalman_estimates(1, i), ...
                                    kalman_estimates(2, i), ...
                                    kalman_estimates(3, i), ...
                                    kp_pos, ki_pos, kd_pos, e_prev_pos, e_int_pos, dt);
        
        % Combine control outputs (consider only vertical thrust from altitude controller)
        u = [u_pos(1); u_pos(2); u_pos(3); 0];  % Assuming no yaw control for simplicity

        % Simulate drone response with updated control input
        [x(i), y(i), z(i), roll(i), pitch(i), yaw(i)] = simulate_drone_dynamics(x(i), y(i), z(i), u, dt);

        % Limit the altitude using the altitude_limiter function
        fprintf('Position: x = %.4f, y = %.4f, z = %.4f\n',  x(i), y(i), z(i));
        u = altitude_limiter(z(i), u);  % Adjust the vertical control if necessary

        % Simulate drone response again after applying altitude limit
        [x(i), y(i), z(i), roll(i), pitch(i), yaw(i)] = simulate_drone_dynamics(x(i), y(i), z(i), u, dt);
        fprintf('u = %.4f  ', u);
        fprintf('\nroll = %.4f, pitch = %.4f, yaw = %.4f', roll(i), pitch(i), yaw(i));
        fprintf('\nFinal Position: x = %.4f, y = %.4f, z = %.4f\n',  x(i), y(i), z(i));
        position_error(i) = sqrt((x(i) - x_target)^2 + (y(i) - y_target)^2);
        % Store the height of the drone
        drone_height_history(i) = z(i); % Store the updated drone height

        % Print the height after applying thrust
        fprintf('Height: %.4f\n', z(i));

        % Call the animation function to visualize the drone's position
        [force_sensor_reading, deformation] = drone_Animation(x, y, z, roll, pitch, yaw, x_desired, y_desired, z_desired, b, z(i), kalman_estimates, X_surface, Y_surface, Z_surface, i); 
        % Store deformation in the history array
        deformation_history(i) = deformation;
        force_history(i) = force_sensor_reading;
        pause(dt);  % Pause for the time step duration
        z_thrust_arr(i) = z(i);
        i = i + 1; % Increment the index
    end
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



function surface_z = get_surface_height(X_surface, Y_surface, Z_surface, x, y)
    % Interpolating the surface height using the input grid and points
    surface_z = interp2(X_surface, Y_surface, Z_surface, x, y, 'linear', NaN);
end

function [X_surface, Y_surface, Z_surface] = initialize_surface()
    % Initialize the surface
    [X_surface, Y_surface] = meshgrid(linspace(-5, 5, 25));
    Z_surface = zeros(size(X_surface)); % Flat surface at z = 0
end

function [force_sensor_reading, deformation] = drone_Animation(x, y, z, roll, pitch, yaw, x_desired, y_desired, z_desired, b, z_thrust, kalman_estimates, X_surface, Y_surface, Z_surface, i)
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
    plot3([x(1), x_desired], [y(1), y_desired], [z(1)-0.3, z_desired+0.02-0.3], 'k--', 'LineWidth', 2);
    
    % Plot the planar blue surface
    surface(X_surface, Y_surface, Z_surface, 'FaceColor', 'blue', 'FaceAlpha', 0.3);
    
    % Clear previous plot
    cla;

    % Plot the planar blue surface
    surface(X_surface, Y_surface, Z_surface, 'FaceColor', 'blue', 'FaceAlpha', 0.3);
    
    % Plot the trajectory with adjusted height
    plot3([x(1), x_desired], [y(1), y_desired], [z(1)-0.3, z_desired+0.02-0.3], 'k--', 'LineWidth', 2);
    
   
    % Plot the drone at the estimated position
    plot_drone(x(i), y(i), z(i), roll(i), pitch(i), yaw(i), b, H, H_m, r_p);

    % Plot the probe at the desired position
    [force_sensor_reading, deformation] = plot_probe(x(i), y(i), z_thrust, X_surface, Y_surface, Z_surface);
    
   % Amplify deformation for visibility
    scaling_factor = 20; % Adjust this value to make the deformation more visible

    deformation_area_radius = 0.3; % Radius around the probe where deformation is applied

    % Loop through surface grid points
    for j = 1:size(X_surface, 1)
        for k = 1:size(Y_surface, 2)
            % Calculate distance from the probe to the current surface grid point
            dist_to_probe = sqrt((X_surface(j, k) - x(i))^2 + (Y_surface(j, k) - y(i))^2);

            % Apply deformation only within the circular area around the probe
            if dist_to_probe <= deformation_area_radius
                % Scale deformation for visibility
                Z_surface(j, k) = Z_surface(j, k) - scaling_factor * deformation;
            end
        end
    end

    % Plot the blue surface with updated Z_surface
    surface(X_surface, Y_surface, Z_surface, 'FaceColor', 'blue', 'FaceAlpha', 0.3);
    
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
    Iyy = 0.2; % Moment of inertia about y-axis
    Izz = 0.3; % Moment of inertia about z-axis
    L = 0.5;  % distance from the center of the quadcopter to any of the propellers (m)
    b = 0.1;  % some appropriately dimensioned constant for drag torque
    
    % Extract control inputs
    throttle = u(1);
    roll_in = u(2);
    pitch_in= u(3);
    yaw_in = u(4);
    % Compute total thrust and torques
    total_thrust = throttle * m * g;
    roll_torque = L * total_thrust * roll_in;
    pitch_torque = L * total_thrust * pitch_in;
    yaw_torque = b * (roll_in^2 - pitch_in^2 + yaw_in^2);
    
    
    % Update linear accelerations (using Newton-Euler equations)
    z_acceleration = -g + total_thrust / m;
    x_acceleration = total_thrust / m * (cos(yaw_in) * sin(pitch_in) * cos(roll_in) + sin(yaw_in) * sin(roll_in));
    y_acceleration = total_thrust / m * (sin(yaw_in) * sin(pitch_in) * cos(roll_in) - cos(yaw_in) * sin(roll_in));
    % Update angular accelerations (using Euler-Lagrange equations)
    p = roll_in;
    q = pitch_in;
    r = yaw_in;
    
    roll_acceleration = (pitch_torque - Izz * q * r) / Ixx;
    pitch_acceleration = (roll_torque - Ixx * p * r) / Iyy;
    yaw_acceleration = (yaw_torque - Iyy * p * q) / Izz;

    % Update positions and angles using Euler integration
    x_new = x + dt * cos(pitch_in) * cos(yaw_in) * cos(roll_in) * dt * x_acceleration;
    y_new = y + dt * (sin(roll_in) * sin(pitch_in) * cos(yaw_in) - cos(roll_in) * sin(yaw_in)) * dt * y_acceleration;
    z_new = z + dt * (cos(roll_in) * sin(pitch_in) * cos(yaw_in) + sin(roll_in) * sin(yaw_in)) * dt * z_acceleration;
    roll_new = roll_in + dt * roll_acceleration;
    pitch_new = pitch_in + dt * pitch_acceleration;
    yaw_new = yaw_in + dt * yaw_acceleration;
end

function [u, e_prev, e_int] = pid_controller(x_target, y_target, z_target, x, y, z, kp, ki, kd, e_prev, e_int, dt)
    % PID Error calculations
    e = [x_target - x; y_target - y; z_target - z]; % Current error
    e_diff = (e - e_prev) / dt; % Derivative of error
    e_int = e_int + e * dt; % Integral of error

    % PID Controller
    u = kp .* e + ki .* e_int + kd .* e_diff; % Control input calculation

    % Update previous error
    e_prev = e;
end

function u = altitude_limiter(z, u)
    z_threshold = 0.3;  % Maximum allowable height

    % Check if the drone exceeds the height threshold
    if z >= z_threshold
        % If the drone is at or above the threshold and the control is pushing upwards
        u(3) = -(1.5*u(3));  % Stop any upward thrust
    end
    if z <= 0.297
        u(3) = -(0.5*u(3));  
    end
    
end

function [force_sensor_reading, deformation]=plot_probe(x, y, z, X_surface, Y_surface, Z_surface)
    % Plot the probe and display force reading if contact is detected.
    
    probe_length = 0.3; % Adjust the length of the probe as needed
    probe_radius = 0.01; % Adjust the radius of the probe as needed
    probe_segments = 20; % Number of segments for the probe cylinder
    
    % Plot the probe cylinder
    [xcylinder_probe, ycylinder_probe, zcylinder_probe] = cylinder(probe_radius, probe_segments);
    surface(xcylinder_probe + x, ycylinder_probe + y, -zcylinder_probe * probe_length + z, 'facecolor', 'k');
    
    [force_sensor_reading, deformation] = calculate_force(x, y, 0.3-z, X_surface, Y_surface, Z_surface);

    
    % Display force sensor reading only if contact force is non-zero
    
    text(x + 0.1, y + 0.1, z + 3, ['      Force(N): ', num2str(norm(force_sensor_reading), '%.4f')], ...
            'FontSize', 18, 'FontWeight', 'bold', 'Color', 'red');
    fprintf('Force = %4f', force_sensor_reading);
end


function plot_drone(x, y, z, ~, ~, ~, b, H, H_m, r_p)
    % Plot the drone at the given position and orientation
    % Define design parameters
    D2R = pi / 180;
    ro = 45 * D2R;                   % angle by which rotate the base of quadcopter
    Ri = [cos(ro), -sin(ro), 0;
          sin(ro), cos(ro),  0;
          0,       0,       1];     % rotation matrix to rotate the coordinates of base 
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
    % Calculate force exerted by the probe on the surface based on contact
    % position of the probe and surface height.
    
    % Get the surface height at the current (x, y) probe position
    surface_z = get_surface_height(X_surface, Y_surface, Z_surface, x_probe, y_probe);
    
    % Initialize force as zero (no contact)
    force = 0;
    
    % Calculate deformation
    deformation = z_probe - surface_z;
    fprintf('deformation=%.4f\n', deformation);
    
    % If the probe touches the surface or penetrates it
    if deformation > 0.00000
        % Apply force based on deformation using Hooke's law
        k = 1000; % Stiffness constant
        force = k * deformation^1.5; % Apply negative force proportional to deformation
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
    title('Drone Trajectory, Noisy Measurements, and Kalman Filter Estimates');
    legend show; % Display legend to identify each dataset
    % Set the view angle
    view(3); % Default 3D view
end

function plot_force(force_history)
    % Plot force sensor reading over time in a separate figure
    figure;
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


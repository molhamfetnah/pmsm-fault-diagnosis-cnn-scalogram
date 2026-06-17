% EXPORT_FOC_CURRENT Sweep FOC load cases and export phase current as .npy.
% Run FOC_PMSM-main/Motor_script.m first (initializes pmsm/inverter structs),
% then run this. Produces Healthy and Overload current via load setpoints.
%
% Path dependency: npy-matlab (writeNPY).

run(fullfile('FOC_PMSM-main', 'Motor_script.m'));

cases = struct('name', {'Healthy_load1', 'Overload_load1'}, ...
               'load', {0.5, 1.5});            % PU torque setpoint
outDir = fullfile('data', 'raw', 'sim', 'foc');
if ~exist(outDir, 'dir'); mkdir(outDir); end

for c = 1:numel(cases)
    % TODO-FOR-USER: set the load/torque input for this case in the model
    % workspace or via the model's input-case selector before simulating.
    out = sim('FOC_PMSM-main/FOCsimulation.slx');
    ia = out.simout.signals.values(:, 1);       % phase-A current (adjust column)
    writeNPY(ia, fullfile(outDir, [cases(c).name '.npy']));
end
disp('FOC current exported.');

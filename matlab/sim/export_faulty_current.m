% EXPORT_FAULTY_CURRENT Sweep inter-turn fault severity on the Simscape
% FaultyPMSM block and export stator current as .npy per severity.
%
% Prerequisites:
%   1. Place FaultyPMSM.ssc in a +pmsm package folder and run: ssc_build pmsm
%   2. Build a minimal open-loop drive harness model (matlab/sim/fault_harness.slx)
%      that instantiates the FaultyPMSM block. Honor the model limits noted in
%      simscape-pmsm/README.md: use a 1 pole-pair config and a simple open-loop
%      drive (NOT full sensorless FOC, which does not converge).
%   3. npy-matlab (writeNPY) on the path.

sigmas = [0.0 0.05 0.10 0.20];   % shorted-turn ratio; 0 = healthy baseline
outDir = fullfile('data', 'raw', 'sim', 'fault');
if ~exist(outDir, 'dir'); mkdir(outDir); end

for s = sigmas
    % TODO-FOR-USER: set block parameter 'sigma' = s in the harness, then sim.
    out = sim('matlab/sim/fault_harness.slx');
    ia = out.simout.signals.values(:, 1);       % stator current (adjust column)
    klass = "InterTurn";
    if s == 0; klass = "Healthy"; end
    name = sprintf('%s_sigma%02d', klass, round(s * 100));
    writeNPY(ia, fullfile(outDir, [char(name) '.npy']));
end
disp('Faulty current exported.');

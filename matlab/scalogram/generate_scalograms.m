% GENERATE_SCALOGRAMS Batch-render CWT scalograms for every signal in the manifest.
% Reads config.yaml + data/manifest.csv, renders one PNG per signal segment,
% and writes the scalogram_path column back into the manifest.
%
% Path dependencies the user must add before running:
%   - npy-matlab (readNPY)                : read .npy segment files
%   - a YAML reader (yaml_read) OR replace the cfg.* lines with literals.
%   - matlab/scalogram on the MATLAB path : scalogram_from_signal

cfg     = yaml_read('config.yaml');        % or hardcode the few values below
wname   = cfg.wavelet;                      % 'amor'
imgSize = cfg.image_size;                   % 224
manifestPath = cfg.paths.manifest;          % 'data/manifest.csv'
segDir  = fullfile(cfg.paths.raw, 'segments');

M = readtable(manifestPath, 'TextType', 'string');
for i = 1:height(M)
    sid = M.signal_id(i);
    sig = readNPY(fullfile(segDir, sid + ".npy"));
    outPath = fullfile(cfg.paths.scalograms, M.signal_type(i), M.class(i), sid + ".png");
    scalogram_from_signal(sig, M.fs(i), wname, imgSize, char(outPath));
    M.scalogram_path(i) = outPath;
end
writetable(M, manifestPath);
disp('Scalograms generated.');

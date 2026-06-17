function scalogram_from_signal(sig, fs, wname, imgSize, outPath)
% SCALOGRAM_FROM_SIGNAL Render the magnitude CWT scalogram of SIG to an
% imgSize x imgSize RGB PNG at OUTPATH.
%   sig     : 1-D signal samples
%   fs      : sampling frequency (Hz)
%   wname   : wavelet name for cwt (e.g. 'amor')
%   imgSize : output square side in pixels (e.g. 224)
%   outPath : destination .png path
    [cfs, ~] = cwt(double(sig(:)), wname, fs);
    A = abs(cfs);
    A = A / max(A(:) + eps);                 % normalize 0..1
    rgb = ind2rgb(im2uint8(A), jet(256));    % colour map -> RGB
    rgb = imresize(rgb, [imgSize imgSize]);
    folder = fileparts(outPath);
    if ~isempty(folder) && ~exist(folder, 'dir')
        mkdir(folder);
    end
    imwrite(rgb, outPath);
end

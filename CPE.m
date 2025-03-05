% channel parameter estimation (CPE)
classdef CPE < handle
    % constants
    properties(Constant)
        OTFS_TYPE_EM = 21;
        OTFS_TYPE_SP = 22;

        PUL_BIORT   = 1;               % bi-orthogonal pulse
        PUL_RECTA   = 2;               % rectangular pulse
        PIL_TYPE_ONE_MIDDLE = 0;       % one pilot in the middle
        PIL_TYPE_LINE_ORTH = 1;        % a line of multiple pilots: pilot, guard, guard, pilot, guard, guard, ...
        PIL_TYPES = [CPE.PIL_TYPE_ONE_MIDDLE, CPE.PIL_TYPE_LINE_ORTH];
    end
    % properties
    properties
        

        pil_type = -1;
    end
end
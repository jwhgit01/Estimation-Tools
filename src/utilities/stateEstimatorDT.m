classdef (Abstract) stateEstimatorDT
%stateEstimatorDT

properties
    % The discrete-time process noise covariance, which may be a constant
    % matrix, an (nv)x(nv)x(N) 3-dimensional array, or the handle of a
    % function whose input is the sample number, k.
    ProcessNoiseCovariance

    % The discrete-time measurement noise covariance, which may be a
    % constant matrix, an (nz)x(nz)x(N) 3-dimensional array, or the handle
    % of a function whose input is the sample number, k.
    MeasurementNoiseCovariance

    % A positive integer indicating how frequent the sample number should
    % be displayed. If zero, sample numbers are not shown.
    DisplayIteration(1,1) {mustBeInteger,mustBeNonnegative} = 0
end

methods (Access=protected)

    function Qk = Q(obj,k)
        %Q Get the process noise covariance at the current sample
        if isa(obj.ProcessNoiseCovariance,'function_handle')
            Qk = obj.ProcessNoiseCovariance(k);
        elseif size(obj.ProcessNoiseCovariance,3) > 1
            Qk = obj.ProcessNoiseCovariance(:,:,k+1);
        else
            Qk = obj.ProcessNoiseCovariance;
        end
    end % Q

    function Rk = R(obj,k)
        %R Get the measurement noise covariance at the current sample
        if isa(obj.MeasurementNoiseCovariance,'function_handle')
            Rk = obj.MeasurementNoiseCovariance(k);
        elseif size(obj.MeasurementNoiseCovariance,3) > 1
            Rk = obj.MeasurementNoiseCovariance(:,:,k+1);
        else
            Rk = obj.MeasurementNoiseCovariance;
        end
    end % R

    function dispIter(obj,k)
        %dispIter Periodically display the iteration number
        if 0 == mod(k+1,obj.DisplayIteration)
            fprintf('k = %i\n',k+1);
        end
    end % dispIter

end % protected methods

end % classdef
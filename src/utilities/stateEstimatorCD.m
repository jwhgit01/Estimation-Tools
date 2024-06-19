classdef (Abstract) stateEstimatorCD
%stateEstimatorCD

properties
    % The discrete-time process noise power soectral density, which may be
    % a constant matrix or the handle of a function whose input is t.
    ProcessNoisePSD

    % The discrete-time measurement noise covariance, which may be a
    % constant matrix, an (nz)x(nz)x(N) 3-dimensional array, or the handle
    % of a function whose input is the sample number, k.
    MeasurementNoiseCovariance

    % The number of intermediate 4th order Runge-Kutta intergation steps
    % between samples.
    nRK(1,1) {mustBeInteger,mustBePositive} = 10

    % A positive number indicating how frequent the sample number should
    % be displayed. If zero, sample times are not shown.
    DisplayPeriod(1,1) {mustBeNonnegative} = 0
end

methods (Access=protected)

    function Qc = Q(obj,t)
        %Q Get the process noise power spectral density at the current time
        if isa(obj.ProcessNoisePSD,'function_handle')
            Qc = obj.ProcessNoisePSD(t);
        else
            Qc = obj.ProcessNoisePSD;
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

    function dispIter(obj,t)
        %dispIter Periodically display the time current time
        if 0 == mod(t,obj.DisplayPeriod)
            fprintf('t = %0.2f\n',t);
        end
    end % dispIter

end % protected methods

end % classdef
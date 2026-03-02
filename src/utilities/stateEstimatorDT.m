classdef (Abstract) stateEstimatorDT
%stateEstimatorDT

properties
    % The discrete-time process noise covariance, which may be a constant
    % matrix, an (nv)x(nv)x(N) 3-dimensional array, or the handle of a
    % function whose input is the sample number, k.
    ProcessNoiseCovariance % Legacy

    % The discrete-time measurement noise covariance, which may be a
    % constant matrix, an (nz)x(nz)x(N) 3-dimensional array, or the handle
    % of a function whose input is the sample number, k.
    MeasurementNoiseCovariance % Legacy

    % A positive integer indicating how frequent the sample number should
    % be displayed. If zero, sample numbers are not shown.
    DisplayIteration(1,1) {mustBeInteger,mustBeNonnegative} = 0
end

methods (Access=protected)
    function dispIter(obj,k)
        %dispIter Periodically display the iteration number
        if 0 == mod(k+1,obj.DisplayIteration)
            fprintf('k = %i\n',k+1);
        end
    end % dispIter
end % protected methods

methods (Static, Access=protected)
    function Ak = makeFun(A)
        %makeFun Make a given system matrix "A" a function of the sample
        % number, k.

        % Function handle
        if isa(A,'function_handle')
            Ak = @(k,varargin) A(k,varargin{:}); % function Ak = A(k,...)
            return;
        end

        % Array
        if isnumeric(A)
            if ndims(A) == 2 %#ok<*ISMAT>
                Ak = @(k,varargin) A; % stationary Ak = A
                return;
            elseif ndims(A) == 3
                Ak = @(k,varargin) A(:,:,k+1); % time-varying, Ak = A(:,:,k+1)
                return;
            end
        end

        % Error handling
        error('stateEstimatorDT:BadSystemMatrix', ...
              '%s must be a matrix, 3-D array, or function handle.',A);
    end % makeFun
end % static, protected methods

end % classdef
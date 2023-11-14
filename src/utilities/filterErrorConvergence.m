function [exitCondition,shouldHalf] = filterErrorConvergence(currentcost,prevcost,iter,khalf,filterErrorParams)

% Default
exitCondition = 0;
shouldHalf = false;

% Relative error
relerror = abs((currentcost-prevcost)/currentcost);
if (relerror < filterErrorParams.RelativeTolerance) && (iter > 0)
    disp('Iteration concluded: relative change in det(R) < tolR')
    exitCondition = 1;
    return
end

% No further improvement
if (currentcost > prevcost) && (khalf >= 10) 
    disp('Error termination:')
    disp('No further improvement after 10 times halving of parameters.')
    exitCondition = 2;
    return
end

% Maximum iterations
if iter == filterErrorParams.MaxIterations
    disp('Maximum number of iterations reached.');
    exitCondition = 3;
    return
end

% Step size halving
if (currentcost > prevcost) && (iter > 0)
    disp('Intermediate divergence: halving of parameter step')
    shouldHalf = true;
    return
end

end
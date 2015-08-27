function theta = minimize(f, init_theta)
% MINIMIZE  Find a local minima of a function f, starting at init_theta.
%          f - function to be minimized. f should be of the form:
%              [cost, grad(, hess)] = f(theta)
    tol = 1e-5; % you should stop optimization when the absolute difference
                % in cost between two iterations is less than tol.
                
    maxIter = 1000; % you should alternatively break after maxIter.
    alpha = 0.1;
    alpha_decay = 0.998;
    % Write your solution below. You should use either gradient descent or
    % Newton-Raphson to find the (local) minimum of the function.
    % Our solution is ~10 lines
    
    useNewton = false;
    if ~useNewton
    
    %% BEGIN SOLUTION (GRADIENT DESCENT)
    
    prev_cost = inf;
    theta = init_theta;
    
    for iter = 1:maxIter
        [cost, grad] = f(theta);
        theta = theta - alpha*grad;
        if abs(cost - prev_cost) < tol, break; end
        alpha = alpha*alpha_decay;
        prev_cost = cost;
    end 
  
    %% BEGIN SOLUTION (NEWTON'S METHOD)
    else
    
    prev_cost = inf;
    theta = init_theta;
    
    for iter = 1:maxIter
        [cost, grad, hess] = f(theta);
        theta = theta - hess\grad;
        if abs(cost - prev_cost) < tol, break; end
        prev_cost = cost;
    end
    
    %% END SOLUTION
    end
end

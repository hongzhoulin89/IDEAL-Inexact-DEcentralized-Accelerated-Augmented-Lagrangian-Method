function [x_end]=SVRG(obj_fn, grad_fn, numIter, x0, eta, data, batch_size, T)
x=x0;
for k=1:numIter
    x_bar = x;
    v = grad_fn(x_bar,data);
    fprintf('Iteration k = %d, suboptimality = %.4e \n', k, norm(v)^2)
    for t = 1:T
        data_subset= data(randperm(length(data),batch_size));
        grad_x = grad_fn(x,data_subset);
        grad_bar = grad_fn(x_bar,data_subset);
        x = x - eta*( grad_x - grad_bar+v); 
    end
end
x_end=x;
end

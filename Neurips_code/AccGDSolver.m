function [iter,err,x_end,test_acc]=AccGDSolver(obj_fn,grad_fn,numProcesses,numIter,x0,W,T,L,mu,eta,eta2,beta_in,beta_out,lambda_2,lambda_n,rho,data_ix,x_optim,batch_size,data_full,X_test,label_test,n_test)
err = zeros(1,numIter);
test_acc = zeros(1,numIter);
[m,m1]=size(x0);
w=zeros(m,numProcesses);
gamma=zeros(m,numProcesses);
x=x0;

err(1,1) = mean((sum((x-x_optim).^2)));
Ytest_pred = linear_prediction(X_test,mean(x,2));
test_acc(1,1) =  sum(Ytest_pred == label_test)/n_test;
val=obj_fn(mean(x,2),data_full);
iter=[val];

for k=1:numIter
    xk = x;
    yk = xk;
    bar_yk = yk*W;
    %%%%%%%%%%%%% Inner Iteration %%%%%%%%%%%%%%%%%
    for t=1:T
        xk_temp=xk;
        grad  = zeros(m,numProcesses);
        % Compute local iterates
        for p_ix=1:numProcesses
            y_local = yk(:,p_ix);
            data_local = data_ix{p_ix};
            grad(:,p_ix) = grad_fn(y_local,data_local);
        end
        grad_AL = gamma + grad + rho*bar_yk;
        xk= yk-eta*grad_AL;
        yk = xk + beta_in*(xk - xk_temp);
        bar_yk = yk*W;
        fprintf('out loop k = %d inner loop t = %d inner suboptimality = %.4e \n', k, t, norm(grad_AL,'fro')^2) 
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%% Outer Iteration %%%%%%%%%%%%%%%%%
    x=xk;
    bar_x = x*W;
    w_old=w;
    w = gamma + eta2* bar_x;
    gamma = w + beta_out*(w - w_old);
    

    val=obj_fn(mean(x,2),data_full);
    iter=[iter,val];
    err(1,k+1) = mean((sum((x-x_optim).^2))); 
    Ytest_pred = linear_prediction(X_test,mean(x,2));
    test_acc(1,k+1) =  sum(Ytest_pred == label_test)/n_test;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
x_end=x;
end
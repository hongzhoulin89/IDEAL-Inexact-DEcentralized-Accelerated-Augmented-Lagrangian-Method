function [iter,obj_err,x_end,test_acc]=EXTRA(obj_fn,grad_fn,numProcesses,numIter,W,x0,alpha,data_ix,x_optim,batch_size, data_full,X_test,label_test,n_test)
%alpha=0.4;   
[d1,d2]=size(x0);    
Wt=0.5*(eye(numProcesses)+W);
x=x0;
obj_err(1,1) = mean((sum((x-x_optim).^2))); 
val=obj_fn(mean(x,2),data_full);
Ytest_pred = linear_prediction(X_test,mean(x,2));
test_acc(1,1) =  sum(Ytest_pred == label_test)/n_test;
iter=[val];

%Compute Gradient 
gradient  = zeros(d1,numProcesses);
for p_ix=1:numProcesses
    data_local = data_ix{p_ix};
    gradient(:,p_ix) = grad_fn(x(:,p_ix),data_local);
end
x1=x*W-alpha*gradient;

% Compute Function    
val=obj_fn(mean(x1,2),data_full);
iter=[iter,val];
obj_err(1,2) = mean((sum((x1-x_optim).^2)));
Ytest_pred = linear_prediction(X_test,mean(x1,2));
test_acc(1,2) =  sum(Ytest_pred == label_test)/n_test;

for k=1:numIter
    gradient_temp=gradient;
    gradient  = zeros(d1,numProcesses);
    for p_ix=1:numProcesses
        data_local = data_ix{p_ix};
        gradient(:,p_ix) = grad_fn(x1(:,p_ix),data_local);
    end 
    x2=x1*(eye(numProcesses)+W)-x*Wt-alpha*(gradient-gradient_temp);
    x=x1;
    x1=x2;
    % Compute Function    
    val=obj_fn(mean(x1,2),data_full);
    iter=[iter,val];
    obj_err(1,k+2) =  mean((sum((x1-x_optim).^2))); 
    Ytest_pred = linear_prediction(X_test,mean(x1,2));
    test_acc(1,k+2) =  sum(Ytest_pred == label_test)/n_test;
end
x_end=x1;
end
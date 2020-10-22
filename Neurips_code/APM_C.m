function [iter,err,x_end,test_acc]=APM_C(obj_fn,grad_fn,numProcesses,lambda2,numIter,InnerIter,x0,W,L,mu,data_ix,x_optim,batch_size,data_full,X_test,label_test,n_test)
% W should be positive defitine !
Weig=sort(eig(W),'ascend');
if Weig <=0
    error("W should be P.S.D.")
end

err=zeros(1,numIter);
test_acc = zeros(1,numIter);

err(1,1)=mean(sum((x0-x_optim).^2));
Ytest_pred = linear_prediction(X_test,mean(x0,2));
test_acc(1,1) =  sum(Ytest_pred == label_test)/n_test;
val=obj_fn(mean(x0,2),data_full);
iter=[val];

% Fixed theta
theta_k= @(k) sqrt(mu/L);

% Diminishing vk:
vk= @(k) (1-theta_k(k))^(k+1);
%% Setting Beta_0 
% Fixed
beta_0= @(k) 1;
%Varying

eta=(1-sqrt(1-lambda2^2))/(1+sqrt(1-lambda2^2));
[d,d2]=size(x0);
xk=x0;
xk_1=x0;
zk=zeros(d,numProcesses);

Win=W;
for i=1:numIter-1
    yk= xk+ (L*theta_k(i)-mu)/(L-mu)*(1-theta_k(i-1))/theta_k(i-1)*(xk-xk_1);
    for agent=1:numProcesses
        data_local = data_ix{agent};
        data_subset=data_local;
        zk(:,agent)=yk(:,agent)-1/L*feval(grad_fn,yk(:,agent),data_subset);
        zkt=zk;
        zkt_1=zk;
        for inloop=1:InnerIter
            zkt_new=(1+eta)*zkt*Win-eta*zkt_1;
            zkt_1=zkt;
            zkt=zkt_new;
        end
    end
    xk_new= L*vk(i)*zk+beta_0(i)*zkt;
    xk_new=xk_new/(L*vk(i)+beta_0(i));
    xk_1=xk;
    xk=xk_new;
    err(1,i+1) = mean((sum((xk_new-x_optim).^2)));
    Ytest_pred = linear_prediction(X_test,mean(xk_new,2));
    test_acc(1,i+1) =  sum(Ytest_pred == label_test)/n_test;
    % Objective computation
    val=obj_fn(mean(xk_new,2),data_full);
    iter=[iter,val];

    
end
x_end=xk;
end
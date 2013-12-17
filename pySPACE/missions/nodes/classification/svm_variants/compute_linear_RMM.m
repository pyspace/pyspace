%% compute_RMM: Compute the RMM from training data
function [w,b] = compute_linear_RMM(labels, train_data,C, B)
	Kzz=(train_data'*train_data);
    [Kw,b,alpha,delta,deltas,res] = rmm(Kzz,labels,C,B);
    w=train_data*Kw;
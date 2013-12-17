
% Copyright (c) 2008, Pannagadatta Shivaswamy and Tony Jebara, Columbia University
% All rights reserved.
%
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are met:
%
% 1. Redistributions of source code must retain the above copyright notice, this
%    list of conditions and the following disclaimer.
% 2. Redistributions in binary form must reproduce the above copyright notice,
%    this list of conditions and the following disclaimer in the documentation
%    and/or other materials provided with the distribution.
%
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
% ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
% WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
% DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
% ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
% (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
% LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
% ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
% (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
% SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
%
% The views and conclusions contained in the software and documentation are those
% of the authors and should not be interpreted as representing official policies,
% either expressed or implied, of the FreeBSD Project.


function [w,bias,alpha,delta,deltas,res]=trans_b_rmm_bias_hard(Kzz,y,C,B)
% Input:
% Kzz is the Gram matrix of the training examples
% y   is the vector of training labels
% C   is the trade off between the slack and the margin
% B   is the upper bound on the training predictions

% Output:
% w, lambda : parameters learned from the training. If Ktz is the 
%  kernel between the test and the training examples, the predictions
% are given by Ktz*w + lambda
% alpha, delta, deltas  are the lagrange multipliers on the constraints
% as in the NIPS paper "Relative Margin Machines"

% Modification by dfeess
% make sure Kzz is symmetric and positive semidefinite:
Kzz=Kzz+eye(size(Kzz,1)).*(-min(real(eig(Kzz)))*2);
Kzz=(Kzz+Kzz')/2;

l = length(y);
n = size(Kzz,1);
u = n - l ;

prob.c = [zeros(n,1) ;  -ones(l,1)  ;  B*ones(n*2,1);  ];
prob.blx = [-inf*ones(n,1) ;  zeros(l,1);  zeros(n*2,1); ];
prob.bux = [ inf*ones(n,1) ;  C*ones(l,1);  inf(n*2,1); ];

quad =  [ Kzz    sparse(n,2*n+l); ...
          sparse(2*n+l,3*n+l) ];

prob.a = [ sparse(1,n) y'  -ones(1,n)  ones(1,n) ; ...
-speye(n)  [ diag(y) ; sparse(u,l) ]   -speye(n) speye(n);...
];
prob.blc = [ sparse(n+1,1); ];
prob.buc = [ sparse(n+1,1); ];
param.MSK_DPAR_INTPNT_CO_TOL_PFEAS = 1.0e-12;

% Dual feasibility tolerance for the dual solution
param.MSK_DPAR_INTPNT_CO_TOL_DFEAS = 1.0e-12;

% Relative primal-dual gap tolerance.
param.MSK_DPAR_INTPNT_CO_TOL_REL_GAP = 1.0e-12;
param.MSK_IPAR_INTPNT_NUM_THREADS = 2;
param.MSK_IPAR_LOG = 1;

[prob.qosubi,prob.qosubj,prob.qoval] = find(tril(sparse(double(quad))));
[r,res]=mosekopt('minimize echo(0)',prob,param);

% some debugging: having access to the res variable
% is necessary to access mosek errors
% save('~/Desktop/res.mat','res')

alpha = res.sol.itr.xx(n+1:n+l);
delta = res.sol.itr.xx(n+l+1:l+2*n);
deltas = res.sol.itr.xx(l+1+2*n:l+3*n);
w = res.sol.itr.xx(1:n);
bias = res.sol.itr.suc(1) - res.sol.itr.slc(1);

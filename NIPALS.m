function [T, P, U, Q, W, B, B2] = NIPALS (X,Y,a)
%
% Inputs:
% X     X matrix normaliz
% Y     Y matrix normaliz
% a     low reduction value
% Outputs:
% T     score for X
% P     loading for X
% U     score for Y
% Q     loading for Y
% B     regression coefficients
% B2    regression Matrix

%Initialization
Ej=X;
Fj=Y;
[nX,mX]  =  size(Ej);
[nY,pY]  =  size(Fj);

% a value controll, until j_max=min(m,n)-> rank(X), Pag.81
j_max = min([mX,nX]); 
if a>j_max
    disp('STOPPED: a value not valid')
    return
end
% preallocating variables for speed
T= zeros(nX,a);
P= zeros(mX,a);
U= zeros(nX,a);
Q= zeros(pY,a);
W= zeros(mX,a);
B= zeros(a,a);
B2= zeros(mX,pY);

%max number of itertion to convergence
n_max_it = 1e3;

for j = 1 : a
    tj_prec = Ej(:,1);
    uj = Fj(:,1);
    % compute Lv and SV until converfence
    for i = 1 : n_max_it
        wj = Ej' * uj/ norm(Ej' * uj);
        tj = Ej * wj;
        qj = Fj' * tj/ norm (Fj' * tj);
        uj = Fj * qj;
        if norm(tj_prec - tj) < 10e-10
            %for convergence 10e-10, example pag, 79
            break
        end
        % updatig tj_prec for next iteration
        tj_prec = tj;
    end

    pj = Ej' * tj/(tj' * tj);
    % scaling not necessary (6.15) to (6.17)
    bj = uj'*tj/(tj'*tj);

    % updatig Ej e Fj for next iteration
    Ej = Ej - tj * pj';
    Fj = Fj - bj * tj * qj';

    % save iteration result:
    T(:, j)= tj;
    P(:, j)= pj;
    U(:, j)= uj;
    Q(:, j)= qj;
    W(:, j)= wj;
    B(j,j)= bj;
    B2= W*pinv(P'*W)*pinv(T'*T)*T'*Y;
    
    % check for residual to stop:
    if (norm(Ej) ==0 && norm(Fj) ==0)
        break
    end
end
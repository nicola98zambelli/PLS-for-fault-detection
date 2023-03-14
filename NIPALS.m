
clc
clear
close all
%load IRIS data
Tbl = readtable('./Fisher.xls');
[n,~]=size(Tbl);
p=3; % n° class type (faut)
m=4; % features

%Class1: Virginica, Class2: Versicolor, Class3: Setos
Tbl= sortrows(Tbl,"Type","descend");
Tbl_Sorted=Tbl;
Tbl_Sorted(1:50,:)=Tbl(51:100,:);
Tbl_Sorted(51:100,:)=Tbl(1:50,:);
X=table2array(Tbl_Sorted(:,3:6));

% normalize X
X_mu=mean(X);
X_std=std(X);
X=normalize(X);
% to denormalize Xnorm(i)*std(i) + mu(i)

% create Y n*p
Y=zeros(n,p);
Y(1:50,1)=ones(50,1);
Y(51:100,2)=ones(50,1);
Y(101:150,3)=ones(50,1);

% normalize Y
Y_mu=mean(Y);
Y_std=std(Y);
Y=normalize(Y);
%function [T, P, U, Q, W, B, B2] = PLSI (X,Y)
%
% Inputs:
% X     X matrix normaliz
% Y     Y matrix normaliz
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
j_max = min([mX,nX]); %until j=min(m,n): rank di X PaG 81(penultima riga)
nMaxOuter = 1e3; %max number of itertion to convergence
a=3; % 0 <= a <= j_max
for j = 1 : a
    % choose The column of X has The largest square of sum as T.
    % choose The column of Y has The largest square of sum as U.
    [~,Tnum] =  max(diag(Ej'*Ej));
    [~,Unum] =  max(diag(Fj'*Fj));
    tj_prec = Ej(:,Tnum);
    uj = Fj(:,Unum);
    % iTeraTion for outer modeling
    for i = 1 : nMaxOuter
        wj = Ej' * uj/ norm(Ej' * uj);
        tj = Ej * wj;
        qj = Fj' * tj/ norm (Fj' * tj);
        uj = Fj * qj;
        if norm(tj_prec - tj) < 10e-10
            %for convergence 10e-10 example pag 79
            break
        end
        % updatig tj_prec for next iteration
        tj_prec = tj;

    end
    pj = Ej' * tj/(tj' * tj);
    %scaling not necessary (6.15) to (6.17)
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

Y_hat=X*B2;
%for each row set the higer value to 1 and the others to 0 according to 6.4
for r=1:n
    max_row=max(Y_hat(r,:));
    for c=1:3
        if Y_hat(r,c)==max_row
            Y_hat(r,c)=1;
        else
            Y_hat(r,c)=0;
        end
    end
end

% show performance
disp('_____________ SHOW PERFORMANCE_______________')
disp('Runing Multiclass confusion matrix')
Y_class=zeros(nY,1);
Y_class(51:100)=1;
Y_class(101:150)=2;
Y_hat_class=zeros(nY,1);
for r=1:n
    if Y_hat(r,2)==1
        Y_hat_class(r)=1;
    else if Y_hat(r,3)==1
        Y_hat_class(r)=2;
    end
    end
end
[c_matrix,Result,RefereceResult]= confusion.getMatrix(Y_class,Y_hat_class);

%plot reducion order X in a dimensions
X_a_reduced=X*P;
scatter3(X_a_reduced(1:50,1),X_a_reduced(1:50,2),X_a_reduced(1:50,3),'*');
hold on;
scatter3(X_a_reduced(51:100,1),X_a_reduced(51:100,2),X_a_reduced(51:100,3),'o');
hold on;
scatter3(X_a_reduced(101:150,1),X_a_reduced(101:150,2),X_a_reduced(101:150,3),'x');
grid on
title("PLS order reduction whit a=3")
xlabel("PLS 1 component")
ylabel("PLS 2 component")
zlabel("PLS 3 component")
legend('Versicolor','Verginica','Setosa')
clc
clear
close all
addpath '../'

%% load Bearing data
Tbl_Normal = readtable('../Data/Bearing_data/Normal_bearing.txt', ...
    VariableNamingRule='preserve');
Tbl_InnerR = readtable('../Data/Bearing_data/Inner_fault.txt', ...
    VariableNamingRule='preserve');
Tbl_OuterR = readtable('../Data/Bearing_data/Outer_race_Fault.txt', ...
    VariableNamingRule='preserve');
Tbl_RollingE = readtable('../Data/Bearing_data/Rolling_elements_fault.txt', ...
    VariableNamingRule='preserve');
addpath '../..'/'Progetto ALES'

%% split data into train validation test  (60,20,20)
perc=[0.6 0.2 0.2];
[n_normal,m_normal]=size(Tbl_Normal);
[n_if,~]=size(Tbl_InnerR);
[n_of,~]=size(Tbl_OuterR);
[n_re,~]=size(Tbl_RollingE);

m=m_normal-1;
n=n_normal+n_re+n_of+n_if;

split_norm= floor(perc*n_normal);
split_if= floor(perc*n_if);
split_of= floor(perc*n_of);
split_re= floor(perc*n_re);

%% create X_train matrix
X_N=table2array(Tbl_Normal(:,1:m));
X_IF=table2array(Tbl_InnerR(:,1:m));
X_OF=table2array(Tbl_OuterR(:,1:m));
X_RE=table2array(Tbl_RollingE(:,1:m));
X=[X_N;X_IF;X_OF;X_RE];

% normalize X
X_mu=mean(X);
X_std=std(X);
X=normalize(X);

%reload standardizated matrix
X_N=X(1:n_normal,:);
X_IF=X(n_normal+1:n_normal+n_if,:);
X_OF=X(n_normal+n_if+1:n_normal+n_if+n_of,:);
X_RE=X(n_normal+n_if+n_of+1:n,:);


n_train= split_norm(1)+split_if(1)+split_of(1)+split_re(1);
n_valid= split_norm(2)+split_if(2)+split_of(2)+split_re(2);
n_test= split_norm(3)+split_if(3)+split_of(3)+split_re(3);

X_N_train=X_N(1:split_norm(1),:);
X_N_valid=X_N(split_norm(1)+1:split_norm(1)+split_norm(2),:);
X_N_test=X_N(split_norm(1)+split_norm(2)+1:n_normal,:);

X_IF_train=X_IF(1:split_if(1),:);
X_IF_valid=X_IF(split_if(1)+1:split_if(1)+split_if(2),:);
X_IF_test=X_IF(split_if(1)+split_if(2)+1:n_if,:);

X_OF_train=X_OF(1:split_of(1),:);
X_OF_valid=X_OF(split_of(1)+1:split_of(1)+split_of(2),:);
X_OF_test=X_OF(split_of(1)+split_of(2)+1:n_of,:);

X_RE_train=X_RE(1:split_re(1),:);
X_RE_valid=X_RE(split_re(1)+1:split_re(1)+split_re(2),:);
X_RE_test=X_RE(split_re(1)+split_re(2)+1:n_re,:);

X_train=[X_N_train;X_IF_train;X_OF_train;X_RE_train];
X_valid=[X_N_valid;X_IF_valid;X_OF_valid;X_RE_valid];
X_test=[X_N_test;X_IF_test;X_OF_test;X_RE_test];

%% create Y_train matrix
p=4; % n° of classes (1 healty, 3 faulty)
Y_N=[ones(n_normal,1) zeros(n_normal,p-1)];
Y_IF=[zeros(n_if,1) ones(n_if,1) zeros(n_if,p-2)];
Y_OF=[zeros(n_of,2) ones(n_of,1) zeros(n_of,p-3)];
Y_RE=[zeros(n_re,3) ones(n_re,1)];
Y=[Y_N;Y_IF;Y_OF;Y_RE];

%normalize Y
Y_mu=mean(Y);
Y_std=std(Y);
Y=normalize(Y);

% reload normalizated matrix
Y_N=Y(1:n_normal,:);
Y_IF=Y(n_normal+1:n_normal+n_if,:);
Y_OF=Y(n_normal+n_if+1:n_normal+n_if+n_of,:);
Y_RE=Y(n_normal+n_if+n_of+1:n,:);

Y_N_train=Y_N(1:split_norm(1),:);
Y_N_valid=Y_N(split_norm(1)+1:split_norm(1)+split_norm(2),:);
Y_N_test=Y_N(split_norm(1)+split_norm(2)+1:n_normal,:);

Y_IF_train=Y_IF(1:split_if(1),:);
Y_IF_valid=Y_IF(split_if(1)+1:split_if(1)+split_if(2),:);
Y_IF_test=Y_IF(split_if(1)+split_if(2)+1:n_if,:);

Y_OF_train=Y_OF(1:split_of(1),:);
Y_OF_valid=Y_OF(split_of(1)+1:split_of(1)+split_of(2),:);
Y_OF_test=Y_OF(split_of(1)+split_of(2)+1:n_of,:);

Y_RE_train=Y_RE(1:split_re(1),:);
Y_RE_valid=Y_RE(split_re(1)+1:split_re(1)+split_re(2),:);
Y_RE_test=Y_RE(split_re(1)+split_re(2)+1:n_re,:);

Y_train=[Y_N_train;Y_IF_train;Y_OF_train;Y_RE_train];
Y_valid=[Y_N_valid;Y_IF_valid;Y_OF_valid;Y_RE_valid];
Y_test=[Y_N_test;Y_IF_test;Y_OF_test;Y_RE_test];



%% compute NIPALS whith a from 1 to m (9) and save regressor matrix
B2_a={};
P_a={};
for a=1:m
    [T, P, U, Q, W, B, B2]=NIPALS(X,Y,a);
    B2_a{a}=B2;
    P_a{a}=P;
end

%% plot low domension a=1
X_a_reduced=X*P_a{1};
figure;
scatter(X_a_reduced(1:n_normal,1),zeros(n_normal,1),'+','LineWidth',2);
hold on;
scatter(X_a_reduced((n_normal+1):n_normal+n_if,1), ...
    zeros(n_if,1),'*','LineWidth',2);
hold on;
scatter(X_a_reduced((n_normal+n_if+1):n_normal+n_if+n_of,1), ...
    zeros(n_of,1),'x','LineWidth',2);
hold on;
scatter(X_a_reduced((n_normal+n_if+n_of+1):n,1), ...
    zeros(n_re,1),"o",'LineWidth',2);
grid on

title("PLS order reduction whit a=1")
xlabel("PLS 1 component")
ylabel("")
legend('Normal','inner race Fault','outer race Fault','rolling element Fault')

%% plot low domension a=2
X_a_reduced=X*P_a{2};
figure;
hold off;
scatter(X_a_reduced(1:n_normal,1),X_a_reduced(1:n_normal,2),'+','LineWidth',2);
hold on;
scatter(X_a_reduced((n_normal+1):n_normal+n_if,1), ...
    X_a_reduced((n_normal+1):n_normal+n_if,2),'*','LineWidth',2);
hold on;
scatter(X_a_reduced((n_normal+n_if+1):n_normal+n_if+n_of,1), ...
    X_a_reduced((n_normal+n_if+1):n_normal+n_if+n_of,2),'x','LineWidth',2);
hold on;
scatter(X_a_reduced((n_normal+n_if+n_of+1):n,1), ...
    X_a_reduced((n_normal+n_if+n_of+1):n,2),"o",'LineWidth',2);
grid on

title("PLS order reduction whit a=2")
xlabel("PLS 1 component")
ylabel("PLS 2 component")
legend('Normal','inner race Fault','outer race Fault','rolling element Fault')

%% plot low domension a=3
X_a_reduced=X*P_a{3};
figure;
hold off;
scatter3(X_a_reduced(1:n_normal,1),X_a_reduced(1:n_normal,2), ...
    X_a_reduced(1:n_normal,3),'+','LineWidth',2);
hold on;
scatter3(X_a_reduced((n_normal+1):n_normal+n_if,1), ...
    X_a_reduced((n_normal+1):n_normal+n_if,2), ...
    X_a_reduced((n_normal+1):n_normal+n_if,3),'*','LineWidth',2);
hold on;
scatter3(X_a_reduced((n_normal+n_if+1):n_normal+n_if+n_of,1), ...
    X_a_reduced((n_normal+n_if+1):n_normal+n_if+n_of,2), ...
    X_a_reduced((n_normal+n_if+1):n_normal+n_if+n_of,2),'x','LineWidth',2);
hold on;
scatter3(X_a_reduced((n_normal+n_if+n_of+1):n,1), ...
    X_a_reduced((n_normal+n_if+n_of+1):n,2), ...
    X_a_reduced((n_normal+n_if+n_of+1):n,2),"o",'LineWidth',2);
grid on

title("PLS order reduction whit a=2")
xlabel("PLS 1 component")
ylabel("PLS 2 component")
zlabel("PLS 3 component")
legend('Normal','inner race Fault','outer race Fault','rolling element Fault')

%% Y_hat Validation phase
Y_hat_a={};
for a=1:m
    Y_hat_a{a}=X_valid*B2_a{a};
    %for each row set the higer value to 1 and the others to 0 according to 6.4
    for r=1:n_valid
        max_row=max(Y_hat_a{a}(r,:));
        for c=1:p
            if Y_hat_a{a}(r,c)==max_row
                Y_hat_a{a}(r,c)=1;
            else
                Y_hat_a{a}(r,c)=0;
            end
        end
    end
end

%% show performance
disp('_____________ SHOW PERFORMANCE_______________')
% Y observed vector class
Y_class=[zeros(split_norm(2),1);ones(split_if(2),1);
    2*ones(split_of(2),1);3*ones(split_re(2),1)];
McE_a={}; %Missclass Error
Precision_a={};
for a=1:m
    fprintf('Runing Multiclass confusion VALIDATION matrix a equal to %d\n',a)
    Y_hat_class=zeros(n_valid,1);
    for r=1:n_valid
        if Y_hat_a{a}(r,2)==1
            Y_hat_class(r)=1;
        else if Y_hat_a{a}(r,3)==1
                Y_hat_class(r)=2;
        else if Y_hat_a{a}(r,4)==1
                Y_hat_class(r)=3;
        end
        end
        end
    end
    %dummy estimation for a =1, because otherwise cufusion funct crash
    if a==1 
        Y_hat_class(1)=1;
        Y_hat_class(2)=2;
    end
    [~,Result,~]= confusion.getMatrix(Y_class,Y_hat_class);
    McE_a{a}=Result.Error;
    Precision_a{a}=Result.Precision;
end

%% plot validation result
figure
plot(cell2mat(McE_a),'LineWidth',2)
hold on;
plot(cell2mat(Precision_a),'LineWidth',2)
grid on

title("PLS order reduction whit a=2")
xlabel("a value")
ylabel("results")
legend('Missclassification error','Precision')

%% Test the model Fixing a equal to 6 
% remember to add values ​​below due to rounding
% +2 for normal and outer race fault

Y_hat=X_test*B2_a{6};
    %for each row set the higer value to 1 and the others to 0 according to 6.4
    for r=1:n_test+4
        max_row=max(Y_hat(r,:));
        for c=1:p
            if Y_hat(r,c)==max_row
                Y_hat(r,c)=1;
            else
                Y_hat(r,c)=0;
            end
        end
    end

Y_class_test=[zeros(split_norm(3)+2,1);ones(split_if(3),1);
    2*ones(split_of(3)+2,1);3*ones(split_re(3),1)];

 fprintf('Runing Multiclass confusion TEST matrix a equal to %d\n',6)
    Y_hat_class=zeros(n_test,1);
    for r=1:n_test+4
        if Y_hat(r,2)==1
            Y_hat_class(r)=1;
        else if Y_hat(r,3)==1
                Y_hat_class(r)=2;
        else if Y_hat(r,4)==1
                Y_hat_class(r)=3;
        end
        end
        end
    end
 [c_matrix,Result,RefereceResult]= confusion.getMatrix(Y_class_test,Y_hat_class);

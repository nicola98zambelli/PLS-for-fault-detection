clc
clear
close all
addpath '../'

Tbl = readtable('../Data/Fisher_data/Fisher.xls');
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

% order reduction:

%%
for aa=1:2
    [T{aa}, P{aa}, U{aa}, Q{aa}, W{aa}, B{aa}, B2{aa}]=NIPALS(X,Y,aa+1);


    Y_hat{aa}=X*B2{aa};
    %for each row set the higer value to 1 and the others to 0 according to 6.4
    for r=1:n
    max_row=max(Y_hat{aa}(r,:));
        for c=1:p
            if Y_hat{aa}(r,c)==max_row
                Y_hat{aa}(r,c)=1;
            else
                Y_hat{aa}(r,c)=0;
            end
        end
    end

% show performance
    disp('_____________ SHOW PERFORMANCE_______________')
    disp('Runing Multiclass confusion matrix')
    Y_class{aa}=zeros(n,1);
    Y_class{aa}(51:100)=1;
    Y_class{aa}(101:150)=2;
    Y_hat_class{aa}=zeros(n,1);
    for r=1:n
        if Y_hat{aa}(r,2)==1
            Y_hat_class{aa}(r)=1;
        else if Y_hat{aa}(r,3)==1
            Y_hat_class{aa}(r)=2;
        end
        end
    end
    [c_matrix,Result,RefereceResult]= confusion.getMatrix(Y_class{aa},Y_hat_class{aa});
    X_a_reduced{aa}=X*P{aa};
    %plot reducion order X in a dimensions
    if(aa==2)
    figure
    scatter3(X_a_reduced{aa}(1:50,1),X_a_reduced{aa}(1:50,2),X_a_reduced{aa}(1:50,3),'*');
    hold on;
    scatter3(X_a_reduced{aa}(51:100,1),X_a_reduced{aa}(51:100,2),X_a_reduced{aa}(51:100,3),'o');
    hold on;
    scatter3(X_a_reduced{aa}(101:150,1),X_a_reduced{aa}(101:150,2),X_a_reduced{aa}(101:150,3),'x');
    grid on
    title("PLS order reduction with a=3")
    xlabel("PLS 1 component")
    ylabel("PLS 2 component")
    zlabel("PLS 3 component")
    legend('Versicolor','Verginica','Setosa')
    hold off
    end
    if(aa==1)
    figure
    scatter(X_a_reduced{aa}(1:50,1),X_a_reduced{aa}(1:50,2),'*');
    hold on;
    scatter(X_a_reduced{aa}(51:100,1),X_a_reduced{aa}(51:100,2),'o');
    hold on;
    scatter(X_a_reduced{aa}(101:150,1),X_a_reduced{aa}(101:150,2),'x');
    grid on
    title("PLS order reduction with a=2")
    xlabel("PLS 1 component")
    ylabel("PLS 2 component")
    legend('Versicolor','Verginica','Setosa')
    hold off
    end
end
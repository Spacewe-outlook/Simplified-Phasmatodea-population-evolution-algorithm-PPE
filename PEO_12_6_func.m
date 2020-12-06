function [Gbest,Fit_min,fit_PEO]= PEO_12_6_func(fhd,Dimension,Particle_Number,Max_Gen,VRmin,VRmax,varargin)
% fhd=str2func('cec14_func'); Fnum = 30;

% func_num =30;

% Max_Gen = Max_Gen;
Np = Particle_Number;
Lb = VRmin;
Ub = VRmax;
Dim = Dimension;

sigma = 0.1*(Ub-Lb);    % Mutation Range (Standard Deviation)
TT = sigma;
% 初始化k个最优解
K = floor(log(Np))+1;
L_pha = zeros(K,Dim);
L_Pha_fit = zeros(1,K);


% 初始化生成x
Pha=initialization(Np,Dim,Ub,Lb);
% 计算fitness
fitness = feval(fhd,Pha',varargin{:});
% k个最优解赋值
[~,f_index] = sort(fitness);
% 初始化全局最优
Fit_min = fitness(f_index(1)); % 全局最优值
Gbest = Pha(f_index(1),:);
% 初始化K个局部最优
for i = 1:K
    L_pha(i,:) = Pha(f_index(i),:);
    L_Pha_fit(i) =  fitness(f_index(i));
end
% 初始化趋势种群趋势
A = zeros(Np,Dim);
% 初始化种群数量x 
Px= (1/Np)*ones(1,Np);
% 初始化种群增长率
Pa = 1.1*ones(1,Np);

count = 1;
fit_PEO(count) = Fit_min;
% pic_num = 1;
for t = 2:Max_Gen
    count = count + 1;
    % 计算新的位置
    new_Pha = Pha + A;
    %边界约束
    new_Pha=space_bound(new_Pha,Ub,Lb);
    
    % 计算所有的fitness
    new_fitness = feval(fhd,new_Pha',varargin{:});
    % 计算最小值和最大值，用来自适应差的情况
    new_fit_best = min(new_fitness); new_fit_worst = max(new_fitness);
    % 更新全局最优
    [~,new_f_index] = sort(new_fitness);
    if new_fitness(new_f_index(1)) <= Fit_min
        Fit_min = new_fitness(new_f_index(1));
        Gbest = new_Pha(new_f_index(1),:);
    end
    % 更新局部最优
    for j = 1:K
        if new_fitness(new_f_index(j))<= L_Pha_fit(j)
            L_Pha_fit(j) = new_fitness(new_f_index(j));
            L_pha(j,:) = new_Pha(new_f_index(j),:);
        end
    end
    % 更新解的位置
    for p = 1:Np
        % 位置是否更新
        if new_fitness(p) <= fitness(p)
            % 位置更新
            Pha(p,:) = new_Pha(p,:);
            fitness(p) = new_fitness(p);
            % 更新繁殖速率
            %计算新的种群数量
            Px(p) = Pa(p)*Px(p)*(1-Px(p));
            %计算新的种群演化趋势
            A1 = choosebest(L_pha,Pha(p,:)); % 趋同进化，最近最优
            A1=A1.*0.2;
            A3 = tubian(Dim,sigma);
            A(p,:) =  (1-Px(p)).*A1+ Px(p).*(A(p,:) + A3);
        else
            if rand <= (Px(p))
                Pha(p,:) = new_Pha(p,:);
                fitness(p) = new_fitness(p);
                %计算新的种群数量
                Px(p) = Pa(p)*Px(p)*(1-Px(p));
            end
            A1 = choosebest(L_pha,Pha(p,:)); % 趋同进化，最近最优
            % 随机步长，随机搜索，不再延续之前进化方式 如果没有这个很难在局部区域移动
            A(p,:) = rand(1,Dim).*A1+TT.*randn(1,Dim); % 这个还不错
            TT = TT*0.99;
        end
        
        %计算种群间的竞争与共存
        % 当前种群，随机选另一个种群，是否距离过近，过近就会产生竞争
        % 计算当前种群的生存区域， 判断生存区域是否交叉
        temp_hab = sigma; % 可以再修改
        % 随机选择一个 % 随机选一个其他的
        temp = randperm(Np);
        if temp(1) ~=p, tp = temp(1);
        else 
            tp = temp(2);
        end
        % 是否会出现竞争
        if dist(Pha(p,:),Pha(tp,:)') < temp_hab*((Max_Gen+1-t)/Max_Gen)
            % 出现竞争
            % 计算生存能力
            d_p = Pa(p)*Px(p)*(1-Px(p)-(fitness(tp)/fitness(p))*Px(tp));
%             d_tp = Pa(tp)*Px(tp)*(1-Px(tp)-(fitness(p)/fitness(tp))*Px(p));
            Px(p) = Px(p) + d_p;
            % 竞争产生的趋势
            A(p,:) = A(p,:) + ((fitness(tp)-fitness(p))/fitness(tp)).*(Pha(tp,:)-Pha(p,:));
            
        end

        
        % 约束a的取值
        if Pa(p) <=0.1 || Pa(p) >=4 || Px(p) <=0.001
            % 这个解重置
            Px(p) = (1/Np);
            Pa(p) = 1.1;
            Pha(p,:) = Lb.*ones(1,Dim) + (Ub-Lb).*ones(1,Dim).*rand(1,Dim);
            A(p,:) = zeros(1,Dim);
            fitness(p) = Inf;
        end
    end
    fit_PEO(count) = Fit_min;
    % 生成新的趋势

end     
%       Fit_min      
end
function A3 = tubian(Dim,sigma)
S = floor(rand/(1/Dim));
A3 = zeros(1,Dim);
if S >=1 && S<=Dim
    J = randperm(Dim,S);
    A3(J) = 1;
    A3 = A3.*2.*sigma.*randn(1,Dim);
%     A3 = (S/Dim).*A3;
end

end

function A1 = choosebest(L_pha,newpha)
% 选择离自己最近的局部最优
[K, dim] = size(L_pha);
temp_dsit = zeros(1,K);
for i = 1:K
    temp_dsit(i) = dist(L_pha(i,:),newpha');
end
[~,index]=min(temp_dsit);
% 第一种 直接全部维选取
A1 = (L_pha(index,:)- newpha);
% 第二种 
end
% This function initialize the first population of search agents
function Positions=initialization(SearchAgents_no,dim,ub,lb)

Boundary_no= size(ub,2); % numnber of boundaries

% If the boundaries of all variables are equal and user enter a signle
% number for both ub and lb
if Boundary_no==1
    Positions=rand(SearchAgents_no,dim).*(ub-lb)+lb;
end

% If each variable has a different lb and ub
if Boundary_no>1
    for i=1:dim
        ub_i=ub(i);
        lb_i=lb(i);
        Positions(:,i)=rand(SearchAgents_no,1).*(ub_i-lb_i)+lb_i;
    end
end
end
%This function checks the search space boundaries for agents.
function  X=space_bound(X,up,low)

[N,dim]=size(X);
for i=1:N 
%     %%Agents that go out of the search space, are reinitialized randomly .
    Tp=X(i,:)>up;Tm=X(i,:)<low;X(i,:)=(X(i,:).*(~(Tp+Tm)))+((rand(1,dim).*(up-low)+low).*(Tp+Tm));
%     %%Agents that go out of the search space, are returned to the boundaries.
%         Tp=X(i,:)>up;Tm=X(i,:)<low;X(i,:)=(X(i,:).*(~(Tp+Tm)))+up.*Tp+low.*Tm;

end
end

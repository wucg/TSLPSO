
function  [avgerror,error_std]=TSLPSO(fhd,D,N,M,Max_FES,func_num,Q,boundary,accuracy)
%      fhd: function handles
%        D: dimension
%        N: population size
%        M: maximum number of iterations
%  Max_FES: maximum number of fitness evaluations 
% func_num: function number
%        Q: run times
% boundary: search space boundary
% accuracy: acceptable accuracy
format long;
%------Parameter initialization------------
c1 = 1.49445;
c2 = 1.49445;
gap1=5;
gap2=5;
sigmax=1;
sigmin=0.1;
sig=1;
FE=zeros(1,Q);
minFE=zeros(1,Q);
Error = zeros(1,Q);
interval=2*boundary;
vmax = 0.5*interval;    
vmin = -vmax;
xmax = boundary;    
xmin = -boundary;
convergeIteration = zeros(1,Q);
runtime=zeros(1,Q);
resultTH.Febest = [];
resultTH.Runtime = [];
resultTH.xm = [];
resultTH.fv = [];
resultTH.accept_iter = [];
resultTH.Mean=[];
resultTH.Std=[];
resultTH.min_FEs = [];
x=zeros(N,D);
v=zeros(N,D);
p=zeros(1,N); 
pe=zeros(N,D);
y=zeros(N,D);
if N==20
    n1=8;
elseif N==40
    n1=15;
else
    n1=12;
end

neiNum = floor(N*0.2);
NeighborDis = zeros(n1,N);
Neighbor = zeros(n1,neiNum);
% -----------random creat,based on the clock------------------
stream = RandStream('mt19937ar', 'Seed', sum(100 * clock));
RandStream.setGlobalStream(stream);
for i=1:N
    pc(i)=0.05+0.45*(exp(10*(i-1)/N-1)-1)/(exp(10)-1);
end
for runNum = 1:Q
    FEs=0;
    starttime = cputime;
    wheel = 0;
    accept_iter = 0;
    accept_fes = 0;
    flag=0;
    flag1=zeros(N,1);
    pbestAge1 = zeros(N,1);   
    
    %-----Population Initialization------------
    for i=1:N       
        for j=1:D            
            x(i,j)=xmin+(xmax-xmin)*rand(1,1);              
            v(i,j)=vmin+(vmax-vmin)*rand(1,1);                
        end        
    end   
    pos=x';
    for i=1:N        
        p(i)=feval(fhd,pos(:,i),func_num);
        FEs=FEs+1;
        if i==1      
            febest(runNum,FEs)=p(i);
        else
            febest(runNum,FEs)=min(febest(runNum,FEs-1),p(i));
        end
        y(i,:)=x(i,:); % Pbest
    end       
    [gbest,ind1]=min(p);
    pg=y(ind1,:);
    %% initialize learning exemplars for DL-subpopulation
    bestfit=p;
    for i=1:n1
        pe(i,:)=y(i,:);
        if isequal(y(i,:),pg)
            for neighNum = 1:N            
                NeighborDis(i,neighNum) = pdist([x(i,:);x(neighNum,:)]);
            end
            [B,index2] = sort(NeighborDis(i,:));
            Neighbor(i,:) = index2(2:neiNum+1);
            lbestfit=p(Neighbor(i,1));
            lbest=y(Neighbor(i,1),:);
            for neinum=2:neiNum
                if p(Neighbor(i,neinum))<lbestfit
                    lbestfit=p(Neighbor(i,neinum));
                    lbest=y(Neighbor(i,neinum),:);
                end
            end 
            nbest=lbest;
        else
            nbest=pg;
        end
        for dem=1:D
           curpbest= pe(i,:);
           if curpbest(dem)==nbest(dem)
              continue
           end
           curpbest(dem)=nbest(dem);
           curpbestvalue=feval(fhd,curpbest',func_num);
           FEs=FEs+1;
           febest(runNum,FEs)=min(febest(runNum,FEs-1),curpbestvalue);
           if curpbestvalue<bestfit(i)
              bestfit(i)=curpbestvalue;
              pe(i,dem)=nbest(dem);
           end
        end   
        if isequal(pe(i,:),y(i,:))
            d1=unidrnd(D);
            rmin=min(x(:,d1));
            rmax=max(x(:,d1));
            pe(i,d1)=rand*(rmin+rmax)-pe(d1);                       
       end
    end

    [bestpe,indx]=min(bestfit);
    if bestpe<gbest
        gbest=bestpe;
        pg=pe(indx,:);
    end
    %% initialize learning exemplars for CL-subpopulation
    fri_best=(1:N)'*ones(1,D);
    for i=n1+1:N 
        fri_best(i,:)=i*ones(1,D);
        friend1=randi([n1+1,N],1,D);
        friend2=randi([n1+1,N],1,D);
        friend=(p(friend1)<p(friend2)).*friend1+(p(friend1)>=p(friend2)).*friend2;
        toss=ceil(rand(1,D)-pc(:,i-n1)');
        if toss==ones(1,D)
            temp_index=randperm(D);
            toss(1,temp_index(1))=0;
            clear temp_index;
        end
        fri_best(i,:)=(1-toss).*friend+toss.*fri_best(i,:);
        for d=1:D
            fri_best_pos(i,d)=y(fri_best(i,d),d);
        end
    end

    %% ------loop------------
    t=0;

    while FEs<=Max_FES
        t=t+1;
        c3=0.5+2*(FEs/Max_FES);
        w1=0.9 - 0.5*(FEs/Max_FES);
        w2=0.9 - 0.5*(FEs/Max_FES);
        
       %% update velocity and positision for DL-subpopulation
        for i=1:n1                  
            for dem = 1:D                    
                v(i,dem)=w1*v(i,dem)+c1*rand*(pe(i,dem)-x(i,dem))+c3*rand*(pg(dem)-x(i,dem));   
                v(i,dem)=min(vmax,max(-vmax,v(i,dem)));
                x(i,dem)=x(i,dem)+v(i,dem);
            end
            isInBorder=1;
            for k=1:D
               if x(i,k)<xmin || x(i,k)>xmax
                    isInBorder=0;
                    break;
                end
            end
            if isInBorder==1
                pos=x';
                fitnessValue1 = feval(fhd,pos(:,i),func_num);
                FEs=FEs+1;
                febest(runNum,FEs)=min(febest(runNum,FEs-1),fitnessValue1);                         
                if fitnessValue1<p(i)
                    p(i)=fitnessValue1;
                    y(i,:)=x(i,:);
                    flag1(i)=1;
                    pbestAge1(i)=0;
                else
                    flag1(i)=0; 
                    pbestAge1(i)=pbestAge1(i)+1;
                end
            end                
        end
       %% update velocity and positision for DL-subpopulation
        for i=(n1+1):N
            for dem = 1:D
                v(i,dem)=w2*v(i,dem)+c2*rand*(fri_best_pos(i,dem)-x(i,dem));
                v(i,dem)=min(vmax,max(-vmax,v(i,dem)));
                x(i,dem)=x(i,dem)+v(i,dem);
            end
            isInBorder=1;
            for k=1:D
               if x(i,k)<xmin || x(i,k)>xmax
                    isInBorder=0;
                    break;
                end
            end
            if isInBorder==1
                pos=x';
                fitnessValue2 = feval(fhd,pos(:,i),func_num);
                FEs=FEs+1;
                febest(runNum,FEs)=min(febest(runNum,FEs-1),fitnessValue2);
                if fitnessValue2<p(i)
                    p(i)=fitnessValue2;
                    y(i,:)=x(i,:);
                    pbestAge1(i)=0;
                else
                    pbestAge1(i)=pbestAge1(i)+1;
                end
            end   
        end
        
       %% perform  Gaussian mutation on gbest
        [gbesttmp,ind2]=min(p);
        pg1=y(ind2,:);   
        if gbesttmp<gbest
            pg=pg1;
            gbest=gbesttmp;
            flag=0;
        else
            flag=flag+1;
        end       
        if flag>=gap1
            pt=pg;
            d1=unidrnd(D);
            randdata =2* rand(1,1)-1;
            pt(d1)=pt(d1)+sign(randdata)*(xmax-xmin)*normrnd(0,sig^2);
            pt(find(pt(:)>xmax))=xmax*rand;
            pt(find(pt(:)<xmin))=xmin*rand;
            cv=feval(fhd,pt',func_num);
            FEs=FEs+1;
            febest(runNum,FEs)=min(febest(runNum,FEs-1),cv);
            if cv<gbest
                pg=pt;
                gbest=cv;
                flag=0;       
            end           
        end         
        sig=sigmax-(sigmax-sigmin)*(FEs/Max_FES);
       %% update learning exemplars for CL-subpopulation
        bestfit=p;
        for i=1:n1
            if flag1(i)==1
               pe(i,:)=y(i,:);
               if isequal(y(i,:),pg)
                    for neighNum = 1:N            
                        NeighborDis(i,neighNum) = pdist([x(i,:);x(neighNum,:)]);
                    end
                    [B,index2] = sort(NeighborDis(i,:));
                    Neighbor(i,:) = index2(2:neiNum+1);
                    lbestfit=p(Neighbor(i,1));
                    lbest=y(Neighbor(i,1),:);
                    for neinum=2:neiNum
                        if p(Neighbor(i,neinum))<lbestfit
                            lbestfit=p(Neighbor(i,neinum));
                            lbest=y(Neighbor(i,neinum),:);
                        end
                    end 
                    nbest=lbest;
               else
                    nbest=pg;
               end
               for dem=1:D
                   curpbest= pe(i,:);
                   if curpbest(dem)==nbest(dem)
                      continue
                   end
                   curpbest(dem)=nbest(dem);
                   curpbestvalue=feval(fhd,curpbest',func_num);
                   FEs=FEs+1;
                   febest(runNum,FEs)=min(febest(runNum,FEs-1),curpbestvalue);
                   if curpbestvalue<bestfit(i)
                      bestfit(i)=curpbestvalue;
                      pe(i,dem)=nbest(dem);
                   end
               end
               if isequal(pe(i,:),y(i,:))
                    d1=unidrnd(D);
                    rmin=min(x(:,d1));
                    rmax=max(x(:,d1));
                    pe(i,d1)=rand*(rmin+rmax)-pe(d1);                       
               end
            end
        end
        [bestpe,indx]=min(bestfit);
        if bestpe<gbest
            gbest=bestpe;
            pg=pe(indx,:);
        end
       %% update learning exemplars for DL-subpopulation
        for i=n1+1:N
            if pbestAge1(i)>=gap2
                fri_best(i,:)=i*ones(1,D);
                friend1=randi([n1+1,N],1,D);
                friend2=randi([n1+1,N],1,D);
                friend=(p(friend1)<p(friend2)).*friend1+(p(friend1)>=p(friend2)).*friend2;
                toss=ceil(rand(1,D)-pc(:,i-n1)');
                if toss==ones(1,D)
                    temp_index=randperm(D);
                    toss(1,temp_index(1))=0;
                    clear temp_index;
                end
                fri_best(i,:)=(1-toss).*friend+toss.*fri_best(i,:);
                for d=1:D
                    fri_best_pos(i,d)=y(fri_best(i,d),d);
                end
                pbestAge1(i)=0;
            end
        end

        if wheel==0                
            if gbest<=accuracy                    
                accept_iter = t;  
                accept_fes=FEs;
                wheel = 1;                    
            end
        end    
  
        if FEs>=Max_FES
            endtime = cputime;
            break;
        end
    end
    
    convergeIteration(1,runNum) = accept_iter;       
    error=gbest-(func_num*100);
    Error(1,runNum)=error;
    FE(1,runNum)=FEs;
    minFE(1,runNum)=accept_fes;
    runtime(1,runNum)= endtime-starttime;

end
resultTH.Runtime = runtime;
resultTH.fv = Error;
resultTH.Febest = febest;
resultTH.accept_iter = convergeIteration;
resultTH.min_FEs = minFE;
avgerror=mean(Error);
minerror=min(Error);
maxerror=max(Error);
avgFEs=mean(minFE);
error_std=std(Error);
resultTH.Mean=avgerror;
resultTH.Std=error_std;

 save(['comput result/' num2str(func_num) '/TSLPSOresult.mat'],'resultTH');

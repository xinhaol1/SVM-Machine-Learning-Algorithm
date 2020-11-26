function [z,e] = SVM(x,y,c,N,d)
%This function uses quadratic programming specifically for HW5
%it takes five inputs: x, y, constant c, training data sets N and dimension
%of the feature vector d
%then it generates two vectors z((d+1)x1) and e(Nx1)
Q = eye(d);
Q(N+d+1,N+d+1)=0;
g = [zeros(1,d+1) c*ones(1,N)];
e1 = -ones(N,1);
A1 = -[y.*x y eye(N)];
lb = [-Inf(d+1,1);zeros(N,1)];
lu = [Inf(N+d+1)];
z = quadprog(Q,g,A1,e1,[],[],lb,lu);
e = z(d+2:end);
z = z(1:d+1);

end


%% 设置模型参数
nx=20;ny=20;nz=10;
dx=100;dy=100;dz=100;
h=0;
%% 调用函数，得到灵敏度矩阵 Hax,Hay,Za
% [Hax,Hay,Za]=fun_forward_mag(nx,ny,nz,dx,dy,dz,h);
[Ta]=fun_forward_mag(nx,ny,nz,dx,dy,dz,h);
%% 求总磁场强度灵敏度矩阵 G_T
% % I0=1/2*pi;    %地磁场倾角
% % A=0;  %地磁场偏角
% cosalpha_t=cos(I0)*cos(A);cosbeta_t=cos(I0)*sin(A);cosgama_t=sin(I0);
% G_T=10^2*(Hax*cosalpha_t+Hay*cosbeta_t+Za*cosgama_t);

G_T=10^2*Ta;
%% 建立模型
M=zeros(ny,nx,nz); 
M(9:12,9:12,3:6)=1; %M为磁化强度 M=1A/m
M=reshape(M,nx*ny*nz,1);
d_obs=G_T*M;
d_obs1=reshape(d_obs,ny,nx);
figure(1)
contourf(d_obs1); colorbar
set(get(colorbar,'title'),'string','nT');
colormap('jet');


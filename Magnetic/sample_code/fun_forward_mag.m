% function [Hax,Hay,Za]=fun_forward_mag(nx,ny,nz,dx,dy,dz,h)
function [Ta]=fun_forward_mag(nx,ny,nz,dx,dy,dz,h)
x0=-(nx-1)*dx:dx:(nx-1)*dx;
y0=-(ny-1)*dy:dy:(ny-1)*dy;
z0=-(nz*dz+h):dz:-(dz+h);
Ai=[-dx/2,dx/2];Bj=[-dy/2,dy/2];Ck=[0,dz];
% Aax=zeros(2*nx-1,2*ny-1,nz);
% Aay=zeros(2*nx-1,2*ny-1,nz);
% Aa=zeros(2*nx-1,2*ny-1,nz);
T=zeros(2*nx-1,2*ny-1,nz);
% alpha=pi/2;beta=pi/2;gama=0;
for m=1:2*nx-1
    for n=1:2*ny-1
        for q=1:nz
            for i=1:2
            for j=1:2
                for k=1:2 
                    xi=x0(m)+Ai(i);
                    yj=y0(n)+Bj(j);
                    zk=z0(q)+Ck(k);
                    uijk=(-1)^(i+j+k);
                    rijk= sqrt(xi^2+yj^2+zk^2);
%                     Aax(m,n,q)=uijk*(cos(alpha)*atan(yj*zk/(xi*rijk))...
% 		                        -cos(beta)*log(rijk+zk)-cos(gama)*log(rijk+yj))+Aax(m,n,q);
% 		            Aay(m,n,q)=uijk*(-cos(alpha)*log(rijk+zk)+cos(beta)*atan(xi*zk/(yj*rijk))...
% 		                            -cos(gama)*log(rijk+xi))+Aay(m,n,q);
% 		            Aa(m,n,q)=-uijk*((-cos(alpha)*log(rijk+yj)-cos(beta)*log(rijk+xi))...
%                         +cos(gama)*atan(xi*yj/(zk*rijk)))+Aa(m,n,q);
                     T(m,n,q)=-uijk*atan(xi*yj/(zk*rijk))+T(m,n,q);
                end
            end  
            end
        end
    end
end
% Hax=zeros(nx*ny,nx*ny*nz);
% Hay=zeros(nx*ny,nx*ny*nz);
% Za=zeros(nx*ny,nx*ny*nz);
for x=1:nx*ny
    for y=1:nx*ny*nz
        n=rem(x,ny);
        if n==0;
            n=ny;
        end
        m=(x-n)/ny+1;
        j=rem(rem(y,nx*ny),ny);
        if j==0;
            j=ny;
        end   
        i=rem(y-j,nx*ny)/ny+1;
        k=(y-j-ny*(i-1))/(nx*ny)+1;
%         Hax(x,y)=Aax(nx-m+i,ny-n+j,nz+1-k);
%         Hay(x,y)=Aay(nx-m+i,ny-n+j,nz+1-k);
%         Za(x,y)=Aa(nx-m+i,ny-n+j,nz+1-k);
Ta(x,y)=T(nx-m+i,ny-n+j,nz+1-k);
    end
end


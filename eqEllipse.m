function [centre,theta,s] = eqEllipse(blob)

% Zero and First moments
m00 = mpq(blob,0,0);
m10 = mpq(blob,1,0);
m01 = mpq(blob,0,1);
% Centre of region
uc = m10/m00;
vc = m01/m00;
centre = [uc vc];
% Second Moments
u20 = upq(blob,2,0);
u11 = upq(blob,1,1);
u02 = upq(blob,0,2);
% Covariance Matrix
J = [u20 u11;...
     u11 u02];
% Eigenvalues (Principal Moments)
lambda = eig(J);
% Major and minor axes
major = 2*sqrt(lambda(2)/m00);
minor = 2*sqrt(lambda(1)/m00);
% Orientation
theta = (0.5)*atan2(2*u11,u20-u02);

% Find the ellipse and the feature points
phase = 0;

t = 0:0.1:2*pi;
x = major*cos(t-phase*pi/180);
y = minor*sin(t-phase*pi/180);

% figure;plot(x,y);axis equal;
[th,r]=cart2pol(x,y);
[x,y]=pol2cart(th+theta,r);
x = x+centre(1);
y = y+centre(2);
figure;
idisp(blob);
if ~(ishold)
    hold on
end
plot(x,y,'r');axis equal;
plot(centre(1),centre(2),'k*');
k = 0:3;
dx = major*cos(k*pi/2);
dy = minor*sin(k*pi/2);
[the2,ro]=cart2pol(dx,dy);
[xx,yy] = pol2cart(the2 + theta,ro); 
xx = xx+centre(1);
yy = yy+centre(2);
s = [xx;yy];
for k = 1:4
    plot(s(1,k),s(2,k),'bo');
end
 hold off;
end

   
       






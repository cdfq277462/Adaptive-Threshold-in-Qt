clear;clc;

% fd = iread('church.png','grey','double');
% fd = iread('lena.png','grey','double');
% fd = iread('street.png','grey','double');
% fd = iread('castle.png','grey','double');
fd = iread('castle2.png','grey','double');
% fd = iread('greenscreen.jpg','grey','double');
% fd = iread('flowers5.png','grey','double');
% fd = iread('monalisa.png','grey','double');

G = 1.3;
k = -0.6;
W = 100;

t = G*niblack(fd,k,W);
T = mean2(t);
count = 0;
err = 0.01;
done = false;
while ~done
    count = count+1
    gd = fd>T;
    Tnext = (0.5)*(mean(fd(gd)) + mean(fd(~gd)));    
    done = abs(T-Tnext) < err;
    T = Tnext          
end

g2 = fd >= t;
gd = fd >= T;
S = kcircle(3);
% g2 = imorph(g2,S,'max');
gd = (gd)&(g2);
closed = iclose(gd,S); 
gd = iopen(closed,S);
figure;idisp(fd);
figure; idisp(gd);

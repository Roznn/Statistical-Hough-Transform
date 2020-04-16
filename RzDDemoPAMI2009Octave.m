% Author: Rozenn Dahyot, email Rozenn.Dahyot@tcd.ie
% Reference PAMI2009: Statistical Hough Transform, R. Dahyot, in IEEE transactions on Pattern Analysis and Machine Intelligence, pages 1502-1509, Vol. 31, No. 8, August 2009.
% Please cite this PAMI paper if code used.
% Demo is launched by typing RzDDemoPAMI2009Octave in command window of Octave,
% the image losange3.bmp needs to be placed in the same folder

function RzDDemoPAMI2009Octave

clear all; close all;
pkg load image

%% Set parameters
std=1; % Gaussian std in the computation of the derivatives

hx=1;hy=1; % bandwidths spatial positions aka pixel location uncertainty [eq. (21) PAMI 2009 ]

noise=20; % standard deviation of the noise added to the image for testing (for testing: = 1, 20, 100)

%% Read image: convert it to gray (intensity image) and add noise
I=imread('losange3.bmp'); 
I=mean(I,3);
I=I+noise*randn(size(I));
figure;imagesc(I),colormap(gray), title('original image')

%% spatial derivatives
[Ix,Iy]=RzDDerivatives(I, std);
[N, theta, rho]=RzDNthetaAlpha(Ix,Iy);

[Height, Width]=size(Ix);
[X,Y]=RzDSpatialCoordinates(Height, Width);

%% estimation of the standard deviation sigma of the noise on the derivatives Ix and Iy: the position of the first maximum  for the magnitude of the gradient N give an estimate for sigma - 
% this sigma cannot be 0 [see ref 2006] and the minimum value is set to 1   
% Reference: Robust Scale Estimation for the Generalized Gaussian Probability Density Function
% R. Dahyot and S. Wilson, in Advances in Methodology and Statistics, n. 1, Vol. 3, pp 21-37, 2006 

spaceN=[max(1,min(N(:))):(max(N(:))-max(0,min(N(:))))/255:max(N(:))];
H=hist(N(:),spaceN);H=H/sum(H);
[I,J]=max(H);

sigma=spaceN(J);
figure;plot(spaceN,H,'b',spaceN(J),H(J),'r.'),title('histogram of the magnitude of the gradient')

%% uncertainty on theta and rho

vartheta=RzDVarTheta(N,sigma);
varrho=RzDVarRho(theta,vartheta);


%% Hough space
% eq. (24) PAMI 2009
maxRho=sqrt(Width.^2+Height^2)/2;

spaceTheta=[-pi/2:pi/180:pi/2];
spaceRho=[-maxRho:1:maxRho];

% equation (7) PAMI 2009
tic
'Equation (7) PAMI 2009: This is slow (about 36s)'
sht1=RzDHoughTransform(theta,vartheta,spaceTheta,rho,varrho,spaceRho,6);
figure;surf(spaceRho,spaceTheta,sht1),colormap(pink), shading interp, title('SHT1: PAMI eq. (7) using observations ( \Theta_i,\rho_i )')
figure;imagesc(iradon(sht1',-spaceTheta*180/pi,'nearest','Hann')), colormap(gray),title('backprojection SHT1 with inverse Radon transform')

toc

% equation (11) PAMI 2009
'Equation (11) PAMI 2009: This is very slow (about 1100s)'
tic
sht2=RzDSHTXYT(theta,vartheta,spaceTheta,spaceRho,hx,hy);
figure;surf(spaceRho,spaceTheta,sht2),colormap(pink), shading interp, title('SHT2: PAMI eq. (11) using observations ( x_i,y_i, \Theta_i)')
figure;imagesc(iradon(sht2',-spaceTheta*180/pi,'nearest','Hann')), colormap(gray),title('backprojection SHT2 with inverse Radon transform')
toc
% equation (15) PAMI 2009 
'Equation (15) PAMI 2009 : This is  slow (about 30s)'
tic
figure;imagesc(N>3*sigma), colormap(gray), title('Edge pixels')
II=find(N>3*sigma); % selected edge points
sht3=RzDSHTXY(X(II),Y(II),spaceTheta,spaceRho,hx,hy);
figure;surf(spaceRho,spaceTheta,sht3),colormap(pink), shading interp, title('SHT3: PAMI eq. (13) using edges only observations ( x_i,y_i)')
figure;imagesc(iradon(sht3',-spaceTheta*180/pi,'nearest','Hann')), colormap(gray),title('backprojection SHT3 with inverse Radon transform')
toc

%%
function [Ix,Iy]=RzDDerivatives(I, std)
% [Ix,Iy]=RzDDerivatives(I, std)
% compute the Gaussian filtered derivatives
% Rozenn Dahyot 2005

% size of the convolution windows
x=[-5*std:5*std];
y=[-5*std:5*std]'; 
% filter: smoothing gaussian in the x direction
XG= exp(-x.^2/(2*std^2)) / (std*sqrt(2*pi));
% filter: smoothing gaussian in the x direction
YG= exp(-y.^2/(2*std^2)) / (std*sqrt(2*pi));
% filter: derivative gaussian in the x direction
XGx= -x.* exp(-x.^2/(2*std^2)) / (std^3*sqrt(2*pi));
% filter: derivative  gaussian in the x direction
YGy=-y.* exp(-y.^2/(2*std^2)) / (std^3*sqrt(2*pi));

XG=XG/norm(XG);
XGx=XGx/norm(XGx);
YG=YG/norm(YG);
YGy=YGy/norm(YGy);

% s='same';
s='valid';
% Ix : DroG
Ix=conv2(I,XGx,s);
Ix=conv2(Ix,YG,s);
    
% Iy : DroG
Iy=conv2(I,YGy,s);
Iy=conv2(Iy,XG,s);
    
%%
function    [N, theta, alpha]=RzDNthetaAlpha(Ix,Iy)
% [N, theta, alpha]=RzDNthetaAlpha(Ix,Iy)
% Ix, Iy, spatial derivatives of an image
% N=sqrt(Ix.^2+Iy.^2); PAMI equation (2)
% theta=atan2(Iy,Ix);   PAMI equation (2)
% alpha=(X.*Ix+Y.*Iy)./N; == rho  PAMI equation (2)
% origin of the coordinates chosen in the middle of the image
% Author: Rozenn DAHYOT
% Revised: 2005

[Height Width]=size(Ix);

[X,Y]=RzDSpatialCoordinates(Height, Width);

N=sqrt(Ix.^2+Iy.^2);

theta=3*pi*ones(size(N));
I=find(N>0);
theta(I)=atan(Iy(I)./Ix(I));


rho=X.*cos(theta)+Y.*sin(theta);
rho=reshape(rho,size(N));

alpha=rho;

%%
function [X,Y]=RzDSpatialCoordinates(Height, Width)
% function [X,Y]=RzDSpatialCoordinates(Height, Width)
% Create matrix X and Y corresponding to the coordinates of the pixels in
% an image of size Width x Height
% the origin of the coordinates is in the middle of the image.
% Author: Rozenn Dahyot
% Year: 2007

X=ones(Height,1)*[1:Width];
Y=[1:Height]'*ones(1,Width);

X=X-Width/2;
Y=Y-Height/2;
%%
function vartheta=RzDVarTheta(N,sigma)
% function vartheta=RzDVarTheta(N,sigma)
% N: Magnitude of the gradient of an image $N=\sqrt{I_x^2+I_y^2}$
% sigma: standard deviation of the noise on the derivatives
% vartheta: variance of theta, the angle of the gradient PAMI equation (5)
% Author: Rozenn Dahyot
% Date: 2007

vartheta=ones(size(N));

II=find(N>0);
vartheta(II)=sigma^2./(N(II).^2); % preventing dividing by 0
I=find(N==0);
if (~isempty(I))
    vartheta(I)=max(max(vartheta(II)),10); % setting a large variance to get a flat enough prior
end

%% minimum uncertainty about theta set around  the order of (pi/180)^2 
clear I;
I=find(vartheta<(pi/180)^2 ); % to prevent the variance to be too small
vartheta(I)=(pi/180)^2;

%%
function varrho=RzDVarRho(theta,vartheta)
% function varrho=RzDVarRho(theta,vartheta)
% theta: angle of the gradient of an image
% vartheta: variance of theta
% varrho: variance of rho
% PAMI equation (6), with pixel uncertainty equal to 1 in both directions.
% Author: Rozenn Dahyot
% 2007

[Height Width]=size(theta);
[X,Y]=RzDSpatialCoordinates(Height, Width);


varrho=1+(Y.*(cos(theta))-X.*(sin(theta))).^2.*vartheta;

%%
function shtpdf=RzDHoughTransform(theta,vartheta,spaceTheta,rho,varrho,spaceRho,B)
% function shtpdf=RzDHoughTransform(theta,vartheta,spaceTheta,rho,varrho,spaceRho,B)
% shtpdf: statistical hough transform probability density function
% B: number by which the image is splitted  to save memory
% equation (7) in PAMI 2009
% Author: Rozenn Dahyot
% Date: 2007
%

[Height Width]=size(theta);

L=Width*Height;
shtpdf=zeros(length(spaceTheta),length(spaceRho));

for ii=0:B-1

    rrange=[ii*ceil(L/B)+1:(ii+1)*floor(L/B)];
    dist=(ones(length(rrange),1)*spaceTheta-theta(rrange)'*ones(1,length(spaceTheta))).^2;

    temptheta=(vartheta(rrange))'*ones(1,length(spaceTheta));
    hhtheta3= exp(-dist./(2*temptheta))./(sqrt(2*pi*temptheta));

    temprho=varrho(rrange)'*ones(1,length(spaceRho));
    drho=(ones(length(rrange),1)*spaceRho-rho(rrange)'*ones(1,length(spaceRho))).^2;
    hhrho3= exp(-drho./(2*temprho))./(sqrt(2*pi*temprho));


    shtpdf=shtpdf+hhtheta3'*hhrho3;

    % The following is to avoid border effect at -pi/2 and + pi/2
    
    %%%%theta+pi
    dist=(ones(length(rrange),1)*spaceTheta-(theta(rrange)+pi)'*ones(1,length(spaceTheta))).^2;

    temptheta=(vartheta(rrange))'*ones(1,length(spaceTheta));
    hhtheta3= exp(-dist./(2*temptheta))./(sqrt(2*pi*temptheta));

    temprho=varrho(rrange)'*ones(1,length(spaceRho));
    drho=(ones(length(rrange),1)*spaceRho-(-rho(rrange))'*ones(1,length(spaceRho))).^2;
    hhrho3= exp(-drho./(2*temprho))./(sqrt(2*pi*temprho));


    shtpdf=shtpdf+hhtheta3'*hhrho3;

    %%%%theta-pi
    dist=(ones(length(rrange),1)*spaceTheta-(theta(rrange)-pi)'*ones(1,length(spaceTheta))).^2;

    temptheta=(vartheta(rrange))'*ones(1,length(spaceTheta));
    hhtheta3= exp(-dist./(2*temptheta))./(sqrt(2*pi*temptheta));

    temprho=varrho(rrange)'*ones(1,length(spaceRho));
    drho=(ones(length(rrange),1)*spaceRho-(-rho(rrange))'*ones(1,length(spaceRho))).^2;
    hhrho3= exp(-drho./(2*temprho))./(sqrt(2*pi*temprho));


    shtpdf=shtpdf+hhtheta3'*hhrho3;


end

shtpdf=shtpdf/(3*length(theta));

%%
function shtpdf=RzDSHTXYT(theta,vartheta,spaceTheta,spaceRho,sx,sy)
% function
% shtpdf=RzDSHTXYT(theta,vartheta,spaceTheta,spaceRho,sx,sy)
% shtpdf: statistical hough transform probability density function with
% observations x y theta
% PAMI equation (11) with kernel equation (19)
% Author: Rozenn Dahyot
% Date: 2007

[Height Width]=size(theta);
[X,Y]=RzDSpatialCoordinates(Height, Width);

STheta=spaceTheta'*ones(size(spaceRho));
SRho=ones(size(spaceTheta))'*spaceRho;

shtpdf=zeros(length(spaceTheta),length(spaceRho));

c3=(sy^2.*sin(STheta).^2+sx^2.*cos(STheta).^2);
c0=1./sqrt(2*pi*c3);
   

for ii=1:length(X(:))

    xx=X(ii);
    yy=Y(ii);

    c2=(SRho-xx.*cos(STheta)-yy.*sin(STheta)).^2./(2*c3);
    g=c0.*exp(-c2);
    g0=g.*(exp(-(STheta-theta(ii)).^2/(2*vartheta(ii)))+exp(-(STheta-theta(ii)+pi).^2/(2*vartheta(ii)))+exp(-(STheta-theta(ii)-pi).^2/(2*vartheta(ii))))/(3*sqrt(2*pi*vartheta(ii)));


    shtpdf=shtpdf+g0;
end

shtpdf=shtpdf/length(X(:));

%%
function shtpdf=RzDSHTXY(X,Y,spaceTheta,spaceRho,sx,sy)
% function shtpdf=RzDSHTXY(X,Y,spaceTheta,spaceRho,sx,sy)
% shtpdf: statistical hough transform probability density function
%  PAMI equation(13)
% X,Y: position of points
% spaceTheta:  [-pi/2:.01:pi/2]
% spaceRho: [-maxRho:1+maxRho]
% Rozenn Dahyot
% 2007


STheta=spaceTheta'*ones(size(spaceRho));
SRho=ones(size(spaceTheta))'*spaceRho;

shtpdf=zeros(length(spaceTheta),length(spaceRho));

c3=(sy^2.*sin(STheta).^2+sx^2.*cos(STheta).^2);
c0=1./sqrt(2*pi*c3);

for ii=1:length(X(:))
    
    xx=X(ii);
    yy=Y(ii);
    
    c2=(SRho-xx.*cos(STheta)-yy.*sin(STheta)).^2./(2*c3);
    g=c0.*exp(-c2);
    shtpdf=shtpdf+g;
end

shtpdf=shtpdf/(pi*length(X(:)));






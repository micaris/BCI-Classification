%% Load Data
load BCI_2003_datasetIII

X = x_train(:,:,:);
y = y_train;

%% Extract left and right motor imagery signal

left_index = find(y == 1);
right_index = find(y == 2);

X_left = X(:,:,left_index);
X_right = X(:,:,right_index);

%% Take the ensemble of the three channels over 140 trials

c3l = mean(X_left(:,1,:),3);
c4l = mean(X_left(:,2,:),3);
czl = mean(X_left(:,3,:),3);

c3r = mean(X_right(:,1,:),3);
c4r = mean(X_right(:,2,:),3);
czr = mean(X_right(:,3,:),3);

figure
plot(c3l)
hold on
plot(c3r)
% hold on
% plot(czl)
xlabel('Time ms')
ylabel('Amplitude')
legend('Average of ensemble from c3 channel')
%% FFT spectrum analysis 

N = length(c3l);
fs = 128;
fn = 128/2;
F = 0:fn/(N-1):fn ;

xft_c3 = abs(fft(c3l));
xft_c4 = abs(fft(c4l));
xft_cz = abs(fft(czl));
xft_c3 = fftshift(xft_c3);
xft_c4 = fftshift(xft_c4);
xft_cz = fftshift(xft_cz);
figure
plot(F,xft_c3)
hold on 
plot(F,xft_c4)
hold on 
plot(F,xft_cz)
xlabel('frequency/Hz')
ylabel('Magnitude')
legend('c3', 'c4', 'cz')
hold off

%% Design band-pass filter 

Fs = 128;
Fn = 128/2;
wp = [8 25]/Fn;
ws = [5 30]/Fn;
rp = 3;
rs = 30;
[ord , Wn] = buttord(wp, ws,rp,rs);
[num, den] = butter(ord,Wn);
freqz(num, den)

%% Design band-stop filter

wps = [8 25]/Fn;
wss = [13 17]/Fn;
rps = 3;
rss = 30;
[ord1 , Wns] = buttord(wps, wss,rps,rss);
[num1, den1] = butter(ord1,Wns,'stop');
freqz(num1, den1)

%% Filter Ensembles
x3_filt = filter(num, den, c3l);
x3_filt = filter(num1, den1, x3_filt);
x4_filt = filter(num, den, c4l);
x4_filt = filter(num1, den1, x4_filt);
xz_filt = filter(num, den, czl);
xz_filt = filter(num1, den1, xz_filt);


y3_filt = filter(num, den, c3r);
y3_filt = filter(num1, den1, y3_filt);
y4_filt = filter(num, den, c4r);
y4_filt = filter(num1, den1, y4_filt);
yz_filt = filter(num, den, czr);
yz_filt = filter(num1, den1, yz_filt);

%% Plot filtere signals
figure

subplot(1,2,1)
plot(x3_filt)
hold on
plot(c3l)
hold off 
xlabel('Time ms')
ylabel('Amplitude')
legend('filtered c3', 'c3')
title('Right motor signal')

subplot(1,2,2)
plot(y3_filt)
hold on
plot(c3r)
xlabel('Time ms')
ylabel('Amplitude')
legend('filtered c3', 'c3')
title('Right motor signal')
hold off

%% FFT spectrum of filtered signals
x_c3 = abs(fft(x3_filt));
x_c4 = abs(fft(x4_filt));
x_cz = abs(fft(xz_filt));
x_c3 = fftshift(x_c3);
x_c4 = fftshift(x_c4);
x_cz = fftshift(x_cz);

y_c3 = abs(fft(y3_filt));
y_c4 = abs(fft(y4_filt));
y_cz = abs(fft(yz_filt));
y_c3 = fftshift(y_c3);
y_c4 = fftshift(y_c4);
y_cz = fftshift(y_cz);

% 
% figure
% plot(F,x_cz)
% hold on 
% plot(F,y_cz)
% % hold on 
% % plot(F,xft_cz)
% xlabel('frequency/Hz')
% ylabel('Magnitude')
% legend('c3 Filtered')

%% Feature Extraction using PSE welch
segmentLength = 127;
noverlap = 63;

[p3l, w3] = pwelch(x3_filt,segmentLength,noverlap);
[p4l, w4] = pwelch(x4_filt,segmentLength,noverlap);
[pzl, wz] = pwelch(xz_filt,segmentLength,noverlap);

[p3r, w3] = pwelch(y3_filt,segmentLength,noverlap);
[p4r, w4] = pwelch(y4_filt,segmentLength,noverlap);
[pzr, wz] = pwelch(yz_filt,segmentLength,noverlap);

figure
plot(w3,p3l)
hold on 
plot(w3,p3r)
% hold on 
% plot(F,xft_cz)
xlabel('frequency/Hz')
ylabel('Magnitude')
legend('PSE welch of Filtered')
%% FE Implementation

s_l = X_left(:,:,4);
s_r = X_right(:,:,4);

sl_filt = filter(num, den, s_l);
sr_filt = filter(num, den, s_r);

sl_filt = filter(num1, den1, sl_filt);
sr_filt = filter(num1, den1, sr_filt);

pl = welch(sl_filt,segmentLength,noverlap);
pr = welch(sr_filt,segmentLength,noverlap);

figure
subplot(1,2,1)
plot(pl)
xlabel('frequency/Hz')
ylabel('Magnitude')
legend('c3','c4','cz')
title('Left')

subplot(1,2,2)
plot(pr)
xlabel('frequency/Hz')
ylabel('Magnitude')
title('Right')
legend('c3','c4','cz')




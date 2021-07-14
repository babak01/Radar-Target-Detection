clear all
clc;
close all


%% Radar Specifications 
%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Frequency of operation = 77GHz
% Max Range = 200m
% Range Resolution = 1 m
% Max Velocity = 100 m/s
%%%%%%%%%%%%%%%%%%%%%%%%%%%

%speed of light = c
%% User Defined Range and Velocity of target
% *%TODO* :
% define the target's initial position and velocity. Note : Velocity
% remains contant
Init_veloc = -50;
Init_range = 150;
c=3e8;


%% FMCW Waveform Generation

% *%TODO* :
%Design the FMCW waveform by giving the specs of each of its parameters.
% Calculate the Bandwidth (B), Chirp Time (Tchirp) and Slope (slope) of the FMCW
% chirp using the requirements above.

Bsweap = c/2; % c/2*range resolution
Tchirp = 5.5*2*200/c;
slope= Bsweap/Tchirp;

%Operating carrier frequency of Radar 
fc= 77e9;             %carrier freq

                                                          
%The number of chirps in one sequence. Its ideal to have 2^ value for the ease of running the FFT
%for Doppler Estimation. 
Nd=128;                   % #of doppler cells OR #of sent periods % number of chirps

%The number of samples on each chirp. 
Nr=1024;                  %for length of time OR # of range cells

% Timestamp for running the displacement scenario for every sample on each
% chirp
t=linspace(0,Nd*Tchirp,Nr*Nd); %total time for samples


%Creating the vectors for Tx, Rx and Mix based on the total samples input.
Tx=zeros(1,length(t)); %transmitted signal
Rx=zeros(1,length(t)); %received signal
Mix = zeros(1,length(t)); %beat signal

%Similar vectors for range_covered and time delay.
r_t=zeros(1,length(t));
td=zeros(1,length(t));


%% Signal generation and Moving Target simulation
% Running the radar scenario over the time. 

for i=1:length(t)         
    
    
    % *%TODO* :
    %For each time stamp update the Range of the Target for constant velocity. 
    r_t(i) = Init_range + Init_veloc*t(i);
    % *%TODO* :
    %For each time sample we need update the transmitted and
    %received signal. 
    td(i) = 2*r_t(i)/c;
    Tx(i) = cos(2*pi*(fc*t(i)+(slope*(t(i)^2)/2)));
    Rx (i)  = cos(2*pi*(fc*(t(i)-td(i))+(slope*((t(i)-td(i))^2)/2)));
    
    % *%TODO* :
    %Now by mixing the Transmit and Receive generate the beat signal
    %This is done by element wise matrix multiplication of Transmit and
    %Receiver Signal
    Mix(i) = Tx(i) .* Rx(i);
    
end

%% RANGE MEASUREMENT


 % *%TODO* :
%reshape the vector into Nr*Nd array. Nr and Nd here would also define the size of
%Range and Doppler FFT respectively.
Mix_reshape = reshape(Mix, [Nr,Nd]);

 % *%TODO* :
%run the FFT on the beat signal along the range bins dimension (Nr) and
%normalize.
fft_output = fft(Mix_reshape,Nr);
fft_output_norm = fft_output/(Tchirp*Bsweap);
 % *%TODO* :
% Take the absolute value of FFT output
abs_fft = abs(fft_output_norm);
 % *%TODO* :
% Output of FFT is double sided signal, but we are interested in only one side of the spectrum.
% Hence we throw out half of the samples.
abs_fft = abs_fft(1:((Tchirp*Bsweap)/2)+1);

%plotting the range
figure ('Name','Range from First FFT')
subplot(2,1,1)

 % *%TODO* :
 % plot FFT output 
f=Bsweap*(0:((Tchirp*Bsweap)/2))/(Tchirp*Bsweap);
plot(f,abs_fft,'r','LineWidth',2)


R=(Tchirp*f*c)/(2*Bsweap);
subplot(2,1,2)
plot(R,abs_fft,'b','LineWidth',2)
axis ([0 200 0 1]);



%% RANGE DOPPLER RESPONSE
% The 2D FFT implementation is already provided here. This will run a 2DFFT
% on the mixed signal (beat signal) output and generate a range doppler
% map.You will implement CFAR on the generated RDM


% Range Doppler Map Generation.

% The output of the 2D FFT is an image that has reponse in the range and
% doppler FFT bins. So, it is important to convert the axis from bin sizes
% to range and doppler based on their Max values.

Mix=reshape(Mix,[Nr,Nd]);

% 2D FFT using the FFT size for both dimensions.
sig_fft2 = fft2(Mix,Nr,Nd);

% Taking just one side of signal from Range dimension.
sig_fft2 = sig_fft2(1:Nr/2,1:Nd);
sig_fft2 = fftshift (sig_fft2);
RDM = abs(sig_fft2);
RDM = 10*log10(RDM) ;

%use the surf function to plot the output of 2DFFT and to show axis in both
%dimensions
doppler_axis = linspace(-100,100,Nd);
range_axis = linspace(-200,200,Nr/2)*((Nr/2)/400);
figure,surf(doppler_axis,range_axis,RDM);

%% CFAR implementation

%Slide Window through the complete Range Doppler Map

% *%TODO* :
%Select the number of Training Cells in both the dimensions.
%Tr
Tr=12;
Td=5;
% *%TODO* :
%Select the number of Guard Cells in both dimensions around the Cell under 
%test (CUT) for accurate estimation
Gr=8;
Gd=5;
% *%TODO* :
% offset the threshold by SNR value in dB
offset=6;
% *%TODO* :
%Create a vector to store noise_level for each iteration on training cells
level_noise = zeros(1,1);


% *%TODO* :
%design a loop such that it slides the CUT across range doppler map by
%giving margins at the edges for Training and Guard Cells.
%For every iteration sum the signal level within all the training
%cells. To sum convert the value from logarithmic to linear using db2pow
%function. Average the summed values for all of the training
%cells used. After averaging convert it back to logarithimic using pow2db.
%Further add the offset to it to determine the threshold. Next, compare the
%signal under CUT with this threshold. If the CUT level > threshold assign
%it a value of 1, else equate it to 0.


   % Use RDM[x,y] as the matrix from the output of 2D FFT for implementing
   % CFAR
cell_number= (2*Tr+2*Gr+1)*(2*Td+2*Gd+1)-(2*Gr+1)*(2*Gd+1);
signal_cfar = zeros(Nr/2,Nd);

for i = 1:(Nr/2 - (2*Gr+2*Tr))
    for j = 1:(Nd - (2*Gd+2*Td))
        
        signal1 = sum(db2pow(RDM(i:i+2*Tr+2*Gr, j:j+2*Td+2*Gd)),'all');
        signal2 = sum(db2pow(RDM(i+Tr:i+Tr+2*Gr, j+Td:j+Td+2*Gd)),'all');    
        level_noise = signal1 - signal2;
        
        th = level_noise/cell_number;      
        th = pow2db(th) + offset;
        th = db2pow(th);
        
        signal = db2pow(RDM(i+Tr+Gr, j+Td+Gd));
        
        if (signal <= th)
            signal = 0;
        else 
            signal = 1;
        end
        
        signal_cfar(i+Tr+Gr,j+Td+Gd) = signal;  
        
    end
end


% *%TODO* :
% The process above will generate a thresholded block, which is smaller 
%than the Range Doppler Map as the CUT cannot be located at the edges of
%matrix. Hence,few cells will not be thresholded. To keep the map size same
% set those values to 0. 
 

%signal(signal~=0 & signal~=1)=0;






% *%TODO* :
%display the CFAR output using the Surf function like we did for Range
%Doppler Response output.
figure,surf(doppler_axis,range_axis,signal_cfar);
colorbar;
title( 'CA-CFAR Filtered RDM surface plot');
xlabel('Speed');
ylabel('Range');
zlabel('Normalized Amplitude');

 
 
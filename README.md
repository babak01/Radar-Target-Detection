# Radar Target Generation and Detection
<br>
<br>
This code is part of one of the projects in [Udacity sensor fusion nanodegree program](https://www.udacity.com/course/sensor-fusion-engineer-nanodegree--nd313). The goal is to detect target using radar sensor. Please keep in mind that the code is in MATLAB and we generate a virtual target as the real data is not available.
<br>
<br>


### Workflow Steps
***
![image11](https://user-images.githubusercontent.com/54375769/125683641-a631e5d2-a832-4bab-bc21-8b3b0a76655b.png)

<br>
<br>

- Configure the FMCW waveform based on the system requirements.
- Define the range and velocity of target and simulate its displacement.
- For the same simulation loop process the transmit and receive signal to determine the beat signal
- Perform Range FFT on the received signal to determine the Range
- Towards the end, perform the CFAR processing on the output of 2nd FFT to display the target.
<br>
<br>

### Radar System Requirments
***
![image14](https://user-images.githubusercontent.com/54375769/125683987-e5022629-f73d-4c00-8355-9281ede750ad.png)


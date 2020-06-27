import numpy as np
import matplotlib.pyplot as plt
import colorednoise as cn

def generate_pam_4(amount_symbols, samples_per_symbol):
    symbols = [-3,-1,1,3]
    signal = np.random.choice(symbols, amount_symbols)
    signal = signal.repeat(samples_per_symbol)
    return signal

def generate_gaussian_noise(m,s,lenght):
    noise = np.random.normal(m,s,lenght)
    return noise

def generate_PSD_noise(s,fmin,length):
    noise = cn.powerlaw_psd_gaussian(s, length, fmin)
    return noise

def decode_pam_4(sample):
    symbols = [-3,-1,1,3]
    error = np.array(symbols)-np.full((len(symbols)),sample)
    abs_error = np.absolute(error)
    decoded = float(symbols[abs_error.argmin()])
    return decoded

# Generate sampled signal plus noise
x1 = generate_pam_4(5000,1) # PAM-4
#N = generate_gaussian_noise(0,0.05,lenght=len(x1)) # WGN
N = generate_PSD_noise(0.1,0.005,length=len(x1))
x = np.add(x1,N) # Add Noise

y = np.zeros_like(x) # FIR output
s = np.zeros_like(x) # Decoder output
z = np.zeros_like(x) # FIR decoded output
h = np.ones_like(x) # Tap evolution
e = np.zeros_like(x) # Error (y-z)

# LMS FIR  ALgorithm
num_err = 0 # Count total symbol errors
mu = 0.0001  # Learning rate
h[0] = 0.8 # Inital tap value
for n in range(1,len(x)):
    y[n] = x[n]+h[n-1]*x[n-1] # FIR
    #s[n] = decode_pam_4(y[n]) # Decoder
    s[n] = x1[n] # Ideal decoding
    if s[n] != x1[n]:
        num_err +=1
    z[n] = s[n]+h[n]*s[n-1] # FIR for decoded symbol
    e[n] = y[n]-z[n] # Error
    h[n] = h[n-1]+e[n]*y[n]*mu # Tap update

# ================================================ #

# Print results
print(f'Decoder errors: {num_err}')

# Print signals
fig, (ax_1,ax_2,ax_3) = plt.subplots(3, 1, sharex=True)

ax_1.set_title('Original signal')
ax_1.margins(0, 0.1)
ax_1.plot(x1,label='X1')
ax_1.plot(x,label='Xn')
ax_1.legend()

ax_2.set_title('Error')
ax_2.plot(e)

ax_3.set_title('Tap value')
ax_3.plot(h)
fig.tight_layout()

plt.show()


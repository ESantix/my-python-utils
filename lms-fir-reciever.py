import numpy as np
import matplotlib.pyplot as plt

def generate_pam_4(amount_symbols, samples_per_symbol):
    symbols = [-3,-1,1,3]
    signal = np.random.choice(symbols, amount_symbols)
    signal = signal.repeat(samples_per_symbol)
    return signal

def generate_gaussian_noise(m,s,lenght):
    noise = np.random.normal(m,s,lenght)
    return noise

def decode_pam_4(sample):
    symbols = [-3,-1,1,3]
    error = np.array(symbols)-np.full((len(symbols)),sample)
    abs_error = np.absolute(error)
    decoded = float(symbols[abs_error.argmin()])
    return decoded

# Generate sampled signal plus noise
x = generate_pam_4(100,3) # PAM-4
n = generate_gaussian_noise(0,1,lenght=len(x)) # WGN
x = np.add(x,n) # Add Noise


y = np.zeros_like(x) # FIR output
s = np.zeros_like(x) # Decoder output
z = np.zeros_like(x) # FIR decoded output
h = np.ones_like(x) # Tap evolution
e = np.zeros_like(x) # Error (y-z)

# LMS FIR  ALgorithm
num_err = 0 # Count total symbol errors
mu = 0.000000001  # Learning rate

h[0] = 0.5
for n in range(1,len(x)):
    y[n] = x[n]+h[n-1]*x[n-1] # FIR
    s[n] = decode_pam_4(y[n]) # Decoder
    z[n] = s[n]+h[n]*s[n-1]

    e[n] = y[n]-z[n]
    h[n] = h[n-1]+e[n]*y[n]*mu


# SHOW SIGNALS
fig, (ax_1,ax_2,ax_3) = plt.subplots(3, 1, sharex=True)


ax_1.set_title('Original signal')
ax_1.margins(0, 0.1)
ax_1.plot(y)

ax_2.set_title('Error')
ax_2.plot(e)

ax_3.set_title('Tap value')
ax_3.plot(h)
fig.tight_layout()

plt.show()


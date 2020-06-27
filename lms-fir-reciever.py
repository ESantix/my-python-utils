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

def decode_pam_4(sample,):
    symbols = [-3,-1,1,3]
    error = np.array(symbols)-np.full((len(symbols)),sample)
    abs_error = np.absolute(error)
    decoded = float(symbols[abs_error.argmin()])
    return decoded

x = generate_pam_4(100,3) # PAM-4
n = generate_gaussian_noise(0,1,lenght=len(x)) # WGN
x = np.add(x,n) # Add Noise

x_old = 1
d_old = 1
h=0.5
mu = 0.001

y = np.zeros_like()
s = np.zeros_like()
z = np.zeros_like()
h = np.zeros_like()
e = np.zeros_like()
num_err = 0
for i in range(sig):
    y = x + h*x_old
    x_old = x.copy()
    
    y.append(fir_out)
    h_n.append(h)
    dec_out = float(decoded(fir_out,symbols))
    dec_out+=h*d_old
    d_old = dec_out

    err = fir_out - dec_out
    err_n.append(np.absolute(err))
    h += err*fir_out*mu


h_n = np.array(h_n)
err_n = np.array(err_n)

# - SHOW SIGNALS
fig, (ax_1,ax_2,ax_fir) = plt.subplots(3, 1, sharex=True)

ax_1.plot(sig)

ax_1.set_title('Original pulse')
ax_1.margins(0, 0.1)
ax_1.plot(sig_clean)
ax_2.plot(err_n)
ax_2.set_title('Error')
ax_fir.plot(h_n)
ax_fir.set_title('Tap value')
fig.tight_layout()
plt.show()


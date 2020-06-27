import numpy as np
import matplotlib.pyplot as plt


symbols = [-3,-1,1,3] # PAM-4
time_per_symbol = 1 #seconds
sample_rate = 1 #samples per second
samples_per_symbol = 1# sample_rate*time_per_symbol #samples per symbol
amount_symbols = 1000 #amount of symbols in signal

sig_clean = np.random.choice(symbols,amount_symbols) #sampled signal with no noise
sig_clean = sig_clean.repeat(samples_per_symbol)

mean = 0
sigma = 0.1
noise = np.random.normal(mean,sigma,len(sig_clean))

sig = np.add(sig_clean, noise)

def decoded(sample,symbols):
    s = np.array(symbols)-np.full((len(symbols)),sample)
    s2 = np.absolute(s)
    d = symbols[s2.argmin()]
    return d 



x_old = 1
d_old = 1
h=0.5
mu = 0.001
y = []
h_n = []
err_n = []
num_err = 0
for x in sig:
    fir_out = x + h*x_old
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


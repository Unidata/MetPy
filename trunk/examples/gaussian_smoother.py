import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 40)
y = np.linspace(-15, 15, 60)
Y,X = np.meshgrid(y,x)
noise = np.random.randn(*X.shape) * 10
data = X**2 + Y**2 + noise
data = np.ma.array(data, mask=((X**2 + Y**2) < 0.4))

data_filt = gaussian_filter(x, y, data, 4, 4)

plt.subplot(1, 2, 1)
plt.imshow(data.T, interpolation='nearest',
    extent=(x.min(), x.max(), y.min(), y.max()))

plt.subplot(1, 2, 2)
plt.imshow(data_filt.T, interpolation='nearest',
    extent=(x.min(), x.max(), y.min(), y.max()))

plt.show()

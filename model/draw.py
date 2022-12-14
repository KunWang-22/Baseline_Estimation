import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# plot traning loss
# training_loss = pd.read_csv("../result/training_loss.csv")
# plt.plot(training_loss["G"], label="Generator loss")
# plt.plot(training_loss["D"], label="Discriminator loss")



real_data = np.load("../result/real_data.npy")
meter_data = np.load("../result/meter_data.npy")
generated_data = np.load("../result/generated_data.npy")

user = 5
# day = 14

plt.plot(real_data[user*90: (user+1)*90, 16*2+1:19*2+1].flatten(), label='real data')
plt.plot(generated_data[user*90: (user+1)*90, 16*2+1:19*2+1].flatten(), label='generated data', alpha=0.8)
plt.plot(meter_data[user*90: (user+1)*90, 16*2+1:19*2+1].flatten(), label='meter data', alpha=0.4)

# plt.plot(real_data.flatten()[user*122*48:48*(user*122+day)], label='real data')
# plt.plot(generated_data.flatten()[user*122*48:48*(user*122+day)], label='generated data', alpha=0.8)
# plt.plot(meter_data.flatten()[user*122*48:48*(user*122+day)], label='meter data', alpha=0.4)

plt.legend()

plt.show()
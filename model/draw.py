import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# plot traning loss
training_loss = pd.read_csv("../result/training_loss.csv")
# plt.plot(training_loss["G"], label="Generator loss")
plt.plot(training_loss["D"], label="Discriminator loss")



real_data = np.load("../result/real_data.npy")
generated_data = np.load("../result/generated_data.npy")
# last_data = np.load("../result.npy")

user = 20
day = 7
# plt.plot(real_data.flatten()[user*122*48:48*(user*122+day)], label='real data')
# plt.plot(generated_data.flatten()[user*122*48:48*(user*122+day)], label='generated data', alpha=0.8)
# plt.plot(last_data.flatten()[:48*10], label='last generated data')

plt.legend()

plt.show()
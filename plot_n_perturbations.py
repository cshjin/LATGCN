import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("output_n_perturbations.csv")
fig = plt.figure(figsize=(5,3))
# x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
x = [1, 2, 3, 4, 5]
y1_cora = [0.36, 0.63, 0.78, 0.83, 0.84]
y2_cora = [0.40, 0.53, 0.64, 0.65, 0.62]
# y1_cora = [0.36, 0.63, 0.78, 0.83, 0.84, 0.84, 0.88, 0.91, 0.89, 0.89]
# y2_cora = [0.40, 0.53, 0.64, 0.65, 0.62, 0.64, 0.58, 0.66, 0.62, 0.63]
# y3_cora = [0.33, 0.64, 0.79, 0.82, 0.84, 0.87, 0.87, 0.89, 0.89, 0.90]
# y4_cora = [0.37, 0.49, 0.65, 0.64, 0.67, 0.66, 0.65, 0.68, 0.64, 0.65]
y1_citeseer = [0.41, 0.55, 0.65, 0.7, 0.79]
y2_citeseer = [0.41, 0.52, 0.56, 0.61, 0.65]
# y3_citeseer = []
# y4_citeseer = []
y1_pubmed = [0.35,0.53,0.67,0.74,0.84]
y2_pubmed = [0.4,0.49,0.58,0.65,0.73]
# y3_pubmed = []
# y4_pubmed = []

# ax = plt.subplot(1,3,1)
# ax.plot(x, y1_cora, '--', c='blue')
# ax.plot(x, y2_cora, c='blue')
# ax.set_title('Cora')
# ax.set_xticks(x)
# ax.set_xticklabels(x)
# ax.set_xlabel("# of allowed perturbations($\Delta$)")
# ax.set_ylabel("Success rate")

ax = plt.subplot(1,2, 1)
ax.plot(x, y1_citeseer, '--', c='blue')
ax.plot(x, y2_citeseer, c='blue')
ax.set_title('Citeseer')
ax.set_xticks(x)
ax.set_xticklabels(x)
ax.set_xlabel("# of allowed perturbations($\Delta$)")
ax.set_ylabel("Success rate")

ax = plt.subplot(1,2, 2)
ax.plot(x, y1_pubmed, '--', c='blue')
ax.plot(x, y2_pubmed, c='blue')
ax.set_title('PubMed')
ax.set_xticks(x)
ax.set_xticklabels(x)
ax.set_xlabel("# of allowed perturbations($\Delta$)")
ax.set_ylabel("Success rate")

# plt.plot(x, y3, '--', c='g')
# plt.plot(x, y4, c='g')
# plt.legend(['GCN-Nettack-A', 'LAT-GCN-Nettack-A', 'GCN-Nettack-AX', 'LAT-GCN-Nettack-AX'], fontsize=8)
# plt.title('Cora')
# plt.xlabel('# of allowed perturbations ($\Delta$)')
# plt.ylabel('Success rate')
fig.legend(['GCN-Nettack-A', 'LAT-GCN-Nettack-A'], ncol=2, loc=8)
plt.tight_layout()
plt.show()

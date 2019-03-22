import matplotlib.pyplot as plt

file = r'G:\PythonProject\Deblur_RemoteSensing\loss_data\loss.txt'
file = open(file)
line = file.readline()
G_GAN = []
G_L1 = []
D = []
while line:
    loss_list = line[:-2].split(' ')
    loss_list = [float(i) for i in loss_list]
    print(loss_list)
    G_GAN.append(loss_list[0])
    G_L1.append(loss_list[1])
    D.append(loss_list[2])
    line = file.readline()

plt.plot(G_GAN)
plt.show()

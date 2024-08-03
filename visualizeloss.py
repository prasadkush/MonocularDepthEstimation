import pickle
import torch
import matplotlib.pyplot as plt
resultsdir = 'results/trial12_CamVid_Dil4/loss_val_list.pkl'
with open(resultsdir, 'rb') as f:
	loss_val_list = pickle.load(f)
print('loss_val_list: ', loss_val_list)
epoch_list = list(range(len(loss_val_list)))
print('epoch_list: ', epoch_list)
#resultsdir = 'results/trial12_CamVid_Dil4/meaniou_val_list.pkl'
#with open(resultsdir, 'rb') as f:
#	meaniou_val_list = pickle.load(f)
#print('meaniou_val_list: ', meaniou_val_list)
resultsdir = 'results/trial12_CamVid_Dil4/training_loss_list.pkl'
with open(resultsdir, 'rb') as f:
	training_loss_list = pickle.load(f)
print('training_loss_list: ', training_loss_list)
epoch_list2 = list(range(len(training_loss_list)))
print('epoch_list2: ', epoch_list2)


plt.plot(epoch_list, loss_val_list)

plt.plot(epoch_list, loss_val_list)
plt.xlabel('epochs')
plt.ylabel('val loss')
# giving a title to my graph
plt.title('val loss vs epochs')
#plt.plot(epoch_list2, meaniou_val_list)
#plt.xlabel('epochs')
#plt.ylabel('mean iou val')
# giving a title to my graph
#plt.title('meaniou val vs epochs')
#plt.plot(epoch_list2, meaniou_val_list)
#plt.xlabel('epochs')
#plt.ylabel('mean iou val')
# giving a title to my graph
#plt.title('meaniou val vs epochs')
plt.plot(epoch_list2, training_loss_list)
plt.xlabel('epochs')
plt.ylabel('training loss')
# giving a title to my graph
plt.title('training loss vs epochs')
#plt.plot(epoch_list2, meaniou_val_list)
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
df = pd.read_excel('ku3.xlsx', sheet_name='Sheet1',na_values=['NA'],encoding = "gbk")
data = df.iloc[:,1:]
train_dataset1 = data.sample(frac=0.7,random_state=0)
test_dataset1 = data.drop(train_dataset1.index)
train_dataset = data.sample(frac=0.8,random_state=0)
test_dataset = data.drop(train_dataset.index)
train_dataset2 = data.sample(frac=0.9,random_state=0)
test_dataset2 = data.drop(train_dataset2.index)
train_stats = train_dataset.describe()
train_stats.pop("Eads(H)")  #去除eads(h)
train_stats = train_stats.transpose()
train_stats1 = train_dataset1.describe()
train_stats1.pop("Eads(H)")  #去除eads(h)
train_stats1 = train_stats1.transpose()
train_stats2 = train_dataset2.describe()
train_stats2.pop("Eads(H)")  #去除eads(h)
train_stats2 = train_stats2.transpose()
train_labels = train_dataset.pop('Eads(H)')
test_labels = test_dataset.pop('Eads(H)')
train_labels1 = train_dataset1.pop('Eads(H)')
test_labels1 = test_dataset1.pop('Eads(H)')
train_labels2 = train_dataset2.pop('Eads(H)')
test_labels2 = test_dataset2.pop('Eads(H)')
def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)
#normed_train_data = np.array(normed_train_data)
#normed_test_data = np.array(normed_test_data)
def norm(x):
  return (x - train_stats1['mean']) / train_stats1['std']
normed_train_data1 = norm(train_dataset1)
normed_test_data1 = norm(test_dataset1)
def norm(x):
  return (x - train_stats2['mean']) / train_stats2['std']
normed_train_data2 = norm(train_dataset2)
normed_test_data2 = norm(test_dataset2)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
def build_model():
  model = keras.Sequential([
    layers.Dense(64,kernel_regularizer=keras.regularizers.l2(0.001), activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(64,kernel_regularizer=keras.regularizers.l2(0.001), activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.Adam(0.001)

  model.compile(optimizer = optimizer,
                loss='mse',
                metrics=['mae', 'mse']
               )
  return model
model = build_model()
model.summary()
example_batch = normed_train_data[:10]  #
example_result = model.predict(example_batch)
array([[0.24784413], [1.2347037 ], [0.24613336], [0.41336268], [0.8393615 ], [0.41500366], [0.09913546], [0.82409847], [0.1479102 ], [0.49127525]], dtype=float32)
# 
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 1000

history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS,validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])
history1 = model.fit(
  normed_train_data1, train_labels1,
  epochs=EPOCHS,validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])
history2 = model.fit(
  normed_train_data2, train_labels2,
  epochs=EPOCHS,validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])
#
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()
#
def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error')
  plt.plot(hist['epoch'], hist['mae'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mae'],
           label = 'Val Error')
  plt.ylim([0,1])
  plt.legend()

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error ')
  plt.plot(hist['epoch'], hist['mse'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mse'],
           label = 'Val Error')
  plt.ylim([0,1])
  plt.legend()
  plt.show()

plot_history(history)
model = build_model()
# 
#keras.callbacks.EarlyStopping
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])

plot_history(history)
history1 = model.fit(normed_train_data1, train_labels1, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])

history2 = model.fit(normed_train_data2, train_labels2, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])

loss1, mae1, mse1 = model.evaluate(normed_test_data1, test_labels1, verbose=2)

print("Testing set Mean Abs Error(MAE): {:5.2f} ".format(mse1))

loss2, mae2, mse2 = model.evaluate(normed_test_data, test_labels, verbose=2)

print("Testing set Mean Abs Error(MAE): {:5.2f} ".format(mse2))

loss3, mae3, mse3 = model.evaluate(normed_test_data2, test_labels2, verbose=2)

print("Testing set Mean Abs Error(MAE): {:5.2f} ".format(mse3))

loss4, mae4, mse4 = model.evaluate(normed_train_data1, train_labels1, verbose=2)

print("Testing set Mean Abs Error(MAE): {:5.2f} ".format(mse4))

loss5, mae5, mse5 = model.evaluate(normed_train_data, train_labels, verbose=2)

print("Testing set Mean Abs Error(MAE): {:5.2f} ".format(mse5))

loss6, mae6, mse6 = model.evaluate(normed_train_data2, train_labels2, verbose=2)

print("Testing set Mean Abs Error(MAE): {:5.2f} ".format(mse6))

from sklearn.metrics import precision_score
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
r2_score(train_labels2, train_predictions2)

r2_score(test_labels1, test_predictions1)

r2_score(train_labels1, train_predictions1)

r2_score(test_labels2, test_predictions2)

r2_score(train_labels, train_predictions)

r2_score(test_labels, test_predictions)

plt.figure(figsize=(6, 6.5))
bwith = 2
ax = plt.gca()#
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)
train_predictions = model.predict(normed_train_data).flatten()
#
_ = plt.plot([-100, 100], [-100, 100],c ='k',linestyle="-",zorder=10,linewidth = bwith)
plt.scatter(train_labels, train_predictions,  alpha=1,s = 100,marker = 'o',c ='r',norm = 1,zorder=30)
#test_labels是y_test
plt.axis('equal')
plt.axis('square')
plt.xlim([-1.5,2])
plt.ylim([-1.5,2])
plt.tick_params(width=2)
plt.yticks(fontproperties = 'Arial', size = 13,FontWeight = 'bold')
plt.xticks(fontproperties = 'Arial', size = 13,FontWeight = 'bold')
plt.text(0.5, -0.95, "Training score: 0.99",fontproperties = 'Arial',FontWeight = 'bold',fontsize = 14)
plt.text(0.5, -1.15, "mse: 0.003",fontproperties = 'Arial',FontWeight = 'bold',fontsize = 14)
plt.text(0.5, -1.35, "mae: 0.037",fontproperties = 'Arial',FontWeight = 'bold',fontsize = 14)
plt.savefig("train80%.png",dpi = 600)
plt.figure(figsize=(6, 6.5))
bwith = 2
ax = plt.gca()#
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)
plt.tick_params(width=2)
test_predictions = model.predict(normed_test_data).flatten()
#
_ = plt.plot([-100, 100], [-100, 100],c ='k',linestyle="-",zorder=10,linewidth = bwith)
plt.scatter(test_labels, test_predictions,  alpha=1,s = 100,marker = 'o',c ='r',norm = 1,zorder=30)
#
plt.axis('equal')
plt.axis('square')
plt.xlim([-1.5,2])
plt.ylim([-1.5,2])
plt.yticks(fontproperties = 'Arial', size = 13,FontWeight = 'bold')
plt.xticks(fontproperties = 'Arial', size = 13,FontWeight = 'bold')
plt.text(0.5, -0.95, "Test score: 0.98",fontproperties = 'Arial',FontWeight = 'bold',fontsize = 14)
plt.text(0.5, -1.15, "mse: 0.007",fontproperties = 'Arial',FontWeight = 'bold',fontsize = 14)
plt.text(0.5, -1.35, "mae: 0.037",fontproperties = 'Arial',FontWeight = 'bold',fontsize = 14)
plt.savefig("test20%.png",dpi = 600)
plt.figure(figsize=(6, 6.5))
train_predictions1 = model.predict(normed_train_data1).flatten()
#
_ = plt.plot([-100, 100], [-100, 100],c ='k',linestyle="-",zorder=10,linewidth = 2)
plt.scatter(train_labels1, train_predictions1,  alpha=1,s = 100,marker = 'o',c ='r',norm = 1,zorder=30)
#
bwith = 2
ax = plt.gca()#
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)
plt.tick_params(width=2)
plt.axis('equal')
plt.axis('square')
plt.xlim([-1.5,2])
plt.ylim([-1.5,2])
plt.yticks(fontproperties = 'Arial', size = 13,FontWeight = 'bold')
plt.xticks(fontproperties = 'Arial', size = 13,FontWeight = 'bold')
plt.text(0.5, -0.95, "Training score: 0.99",fontproperties = 'Arial',FontWeight = 'bold',fontsize = 14)
plt.text(0.5, -1.15, "mse: 0.002",fontproperties = 'Arial',FontWeight = 'bold',fontsize = 14)
plt.text(0.5, -1.35, "mae: 0.036",fontproperties = 'Arial',FontWeight = 'bold',fontsize = 14)
plt.savefig("train70%.png",dpi = 600)
plt.figure(figsize=(6, 6.5))
test_predictions1 = model.predict(normed_test_data1).flatten()
#
_ = plt.plot([-100, 100], [-100, 100],c ='k',linestyle="-",zorder=10,linewidth = 2)
plt.scatter(test_labels1, test_predictions1,  alpha=1,s = 100,marker = 'o',c ='r',norm = 1,zorder=30)
#
bwith = 2
ax = plt.gca()#
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)
plt.tick_params(width=2)
plt.axis('equal')
plt.axis('square')
plt.xlim([-1.5,2])
plt.ylim([-1.5,2])
plt.yticks(fontproperties = 'Arial', size = 13,FontWeight = 'bold')
plt.xticks(fontproperties = 'Arial', size = 13,FontWeight = 'bold')
plt.text(0.5, -0.95, "Test score: 0.98",fontproperties = 'Arial',FontWeight = 'bold',fontsize = 14)
plt.text(0.5, -1.15, "mse: 0.008",fontproperties = 'Arial',FontWeight = 'bold',fontsize = 14)
plt.text(0.5, -1.35, "mae: 0.067",fontproperties = 'Arial',FontWeight = 'bold',fontsize = 14)
plt.savefig("test30%.png",dpi = 600)
plt.figure(figsize=(6, 6.5))
train_predictions2 = model.predict(normed_train_data2).flatten()
#
_ = plt.plot([-100, 100], [-100, 100],c ='k',linestyle="-",zorder=10,linewidth = 2)
plt.scatter(train_labels2, train_predictions2,  alpha=1,s = 100,marker = 'o',c ='r',norm = 1,zorder=30)
#
bwith = 2
ax = plt.gca()#
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)
plt.tick_params(width=2)
plt.axis('equal')
plt.axis('square')
plt.xlim([-1.5,2])
plt.ylim([-1.5,2])
plt.yticks(fontproperties = 'Arial', size = 13,FontWeight = 'bold')
plt.xticks(fontproperties = 'Arial', size = 13,FontWeight = 'bold')
plt.text(0.5, -0.95, "Training score: 0.99",fontproperties = 'Arial',FontWeight = 'bold',fontsize = 14)
plt.text(0.5, -1.15, "mse: 0.003",fontproperties = 'Arial',FontWeight = 'bold',fontsize = 14)
plt.text(0.5, -1.35, "mae: 0.041",fontproperties = 'Arial',FontWeight = 'bold',fontsize = 14)
plt.savefig("train90%.png",dpi = 600)
plt.figure(figsize=(6, 6.5))
test_predictions2 = model.predict(normed_test_data2).flatten()
#
_ = plt.plot([-100, 100], [-100, 100],c ='k',linestyle="-",zorder=10,linewidth = 2)
plt.scatter(test_labels2, test_predictions2,  alpha=1,s = 100,marker = 'o',c ='r',norm = 1,zorder=30)
#
bwith = 2
ax = plt.gca()#
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)
plt.tick_params(width=2)
plt.axis('equal')
plt.axis('square')
plt.xlim([-1.5,2])
plt.ylim([-1.5,2])
plt.yticks(fontproperties = 'Arial', size = 13,FontWeight = 'bold')
plt.xticks(fontproperties = 'Arial', size = 13,FontWeight = 'bold')
plt.text(0.5, -0.95, "Test score: 0.98",fontproperties = 'Arial',FontWeight = 'bold',fontsize = 14)
plt.text(0.5, -1.15, "mse: 0.007",fontproperties = 'Arial',FontWeight = 'bold',fontsize = 14)
plt.text(0.5, -1.35, "mae: 0.064",fontproperties = 'Arial',FontWeight = 'bold',fontsize = 14)
plt.savefig("test10%.png",dpi = 600)




import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns   #記得只能載 0.8.1版本 (pip install seaborn==0.8.1)

class CCF:
    def __init__(self,url,maxlag=0):
#         d=pd.read_excel(url, encoding='latin-1')
        d=url
        self.label=d.iloc[:,0:].keys()     #每個變量的名稱
        # self.label=d[:,0:].keys()
        self.data=np.array(d.iloc[:,0:])   #(2000, 90)
        # self.data=np.array(d[:,0:])
        self.maxlag=maxlag                 # 0
        self.cov=np.zeros([self.data.shape[1],self.data.shape[1]]) #(90,90)

    def cal(self):
        if(self.maxlag==0):
            self.cov=np.corrcoef(self.data.T)  #(90, 90)

        for i in range(self.cov.shape[1]):
            for j in range(self.cov.shape[1]):
                self.cov[i][j]=abs(self.cov[i][j])
                if(np.isnan(self.cov[i][j])):
                    self.cov[i][j]=0.001  #(90, 90) 這裡是把每個相關值都變成正數

    def show(self):  #畫圖(每個變量的相關性)
        label=list(self.label)   #每個變量的名稱
        temp=pd.DataFrame(self.cov,index=label,columns=label)
        temp_=pd.DataFrame(self.cov)
#         print(temp_)
        f, ax = plt.subplots(figsize=(14, 14))
        sns.heatmap(temp, annot=False, square=True,cmap="Blues",fmt='.1f', linewidths=1)  #畫圖設定
        plt.show()

    def get_matrix(self):  # 畫圖(每個變量的相關性)
        label = list(self.label)  # 每個變量的名稱
        temp = pd.DataFrame(self.cov, index=label, columns=label)

        return temp

    # def select(self,index,threshold_lower=0.3,threshold_upper=0.9):  #index為選擇的變量
    #     res=[]
    #     num=self.label.get_loc(index)   #採用函數查找傳遞值的位置
    #     for i in range(0,self.cov.shape[1]):  #0-90
    #         if(i==num):
    #             res.append(self.label[i])
    #         if(abs(self.cov[i,num])>=threshold_lower and abs(self.cov[i,num])<=threshold_upper):
    #             res.append(self.label[i])
    #     return res

import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class reader(object):
	def __init__(self, filename='DXYArea.csv'):
		self.starttime = self.str2timie('2020-01-24 00:00:00.000')
		self.df = pd.read_csv(filename, sep=',')
		self.maxday = self.calIntervalTime(self.df['updateTime'][0])
		self.days = list(range(self.maxday + 1))
	
	def str2timie(self, text):
		return datetime.datetime.strptime(text, '%Y-%m-%d %H:%M:%S.%f')
	
	def calIntervalTime(self, text):
		return (self.str2timie(text) - self.starttime).days
	
	def get(self):
		dic_vis = {}
		self.num = [0] * (self.maxday + 1)
		pname = self.df['provinceName']
		num = self.df['province_confirmedCount']
		num1 = self.df['province_curedCount']
		num2 = self.df['province_deadCount']
		ptime = self.df['updateTime']
		# print(len(pname))
		for idx in range(len(pname)):
			if '湖北' in pname[idx]:
				continue
			
			day = self.calIntervalTime(ptime[idx])
			if pname[idx] not in dic_vis.keys():
				dic_vis[pname[idx]] = []
			
			if day not in dic_vis[pname[idx]]:
				dic_vis[pname[idx]].append(day)
				# self.num[day] += num[idx] + num1[idx] + num2[idx]
				self.num[day] += num[idx]
		
		print('days\t', self.days)
		print('nums\t', self.num)
	
	def draw(self, topnum, addnum, w, bias, mul):
		
		self.get()
		# plot函数作图
		x_data = self.days
		y_data = self.num
		plt.scatter(x_data, y_data, color='k')
		
		x = np.array(list(range(0, 30)))
		y = topnum / (addnum + np.exp(w * x + bias)) * mul
		ii = 0
		for ty in y:
			print(ii, ty)
			ii += 1
		plt.plot(x, y)
		# show函数展示出这个图，如果没有这行代码，则程序完成绘图，但看不到
		plt.show()
	
	def draw2(self, timebias, topnum, addnum, mul):
		
		self.get()
		# plot函数作图
		x_data = self.days
		y_data = self.num
		y_data[11] = y_data[10] + 3235 - 2345
		plt.scatter(x_data, y_data, color='k')
		
		x = np.array(list(range(0, 30)))
		y = (topnum / (addnum + np.exp(-x + timebias))) * mul
		ii = 0
		for ty in y:
			print(ii, ty)
			ii += 1
		plt.plot(x, y)
		# show函数展示出这个图，如果没有这行代码，则程序完成绘图，但看不到
		plt.show()


if __name__ == '__main__':
	r = reader()
	# r.draw(9.6715, 1.2517, -0.4058, 3.0016, 1000)
	# r.draw2(-11.4268, -382.6507, -13.5347, 100)
	# r.draw(0.5919, 0.0766, -0.4060, 0.2086, 1000)
	# r.draw(-0.3990, -0.1300, -7.4968, 0.9021, 1000)
	# r.draw(17.2113, 0.2228, -0.4061, 1.2763, 100)
	# r.draw(3.0748, 0.3795, -0.4030, 1.8423, 1000)  # 2.3
	# r.draw(0.5939, 0.0676, -0.3710, 0.1156, 1000)  # 2.4
	r.draw(1.8373, 0.1990, -0.3571, 1.2006, 1000)  # 2.5

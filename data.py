import matplotlib.pyplot as plt 
import csv 
  
x = [] 
y = [] 
  
with open('data_daily.csv','r') as csvfile: 
    lines = csv.reader(csvfile, delimiter=',') 
    next(lines)
    for row in lines: 
        x.append(row[0]) 
        y.append(int(row[1])) 
  
plt.plot(x, y, color = 'g') 
  
plt.xticks(rotation = 25) 
plt.xlabel('') 
plt.ylabel('Reciept_Count') 
plt.title('Report', fontsize = 20) 
#plt.grid() 
plt.legend() 
plt.show() 
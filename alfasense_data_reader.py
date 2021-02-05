import pandas as pd
from pandas import DataFrame
import numpy as np
import math
import matplotlib.pyplot as plt
import os
dt=[]
data=[]

directory1 = r"C:\Users\MARTA\Documents\Alphasense\2020-03-12\C21B2+PM(10tebervlidan 12 martamde)"
counter1 = 0
for entry in os.scandir(directory1):
    if(counter1==12):
        break
    with open(entry,"rb") as f:
        while (bytes := f.read(81)):
            year=int.from_bytes(bytes[0:2], byteorder='little', signed=False)
            month=int.from_bytes(bytes[2:3], byteorder='little', signed=False)         
            day=int.from_bytes(bytes[3:4], byteorder='little', signed=False)
            hour=int.from_bytes(bytes[4:5], byteorder='little', signed=False)
            minute=int.from_bytes(bytes[5:6], byteorder='little', signed=False)
            second=int.from_bytes(bytes[6:7], byteorder='little', signed=False)
            pressure=int.from_bytes(bytes[7:11], byteorder='little', signed=False)
            temperature=int.from_bytes(bytes[11:15], byteorder='little', signed=False)
            humidity=int.from_bytes(bytes[15:19], byteorder='little', signed=False)
            ADC_CH_0_CO_B4_WE=int.from_bytes(bytes[19:23], byteorder='little', signed=False)
            ADC_CH_0_CO_B4_WE=(ADC_CH_0_CO_B4_WE*2.5)/(2**31)
            ADC_CH_1_CO_B4_AUX=int.from_bytes(bytes[23:27], byteorder='little', signed=False)
            ADC_CH_1_CO_B4_AUX=(ADC_CH_1_CO_B4_AUX*2.5)/(2**31)
            ADC_CH_2_NO_B4_WE=int.from_bytes(bytes[27:31], byteorder='little', signed=False)
            ADC_CH_2_NO_B4_WE=(ADC_CH_2_NO_B4_WE*2.5)/(2**31)
            ADC_CH_3_NO_B4_AUX=int.from_bytes(bytes[31:35], byteorder='little', signed=False)
            ADC_CH_3_NO_B4_AUX=(ADC_CH_3_NO_B4_AUX*2.5)/(2**31)
            ADC_CH_4_NO2_B43F_WE=int.from_bytes(bytes[35:39], byteorder='little', signed=False)
            ADC_CH_4_NO2_B43F_WE=(ADC_CH_4_NO2_B43F_WE*2.5)/(2**31)
            ADC_CH_5_NO2_B43F_AUX=int.from_bytes(bytes[39:43], byteorder='little', signed=False)
            ADC_CH_5_NO2_B43F_AUX=(ADC_CH_5_NO2_B43F_AUX*2.5)/(2**31)
            ADC_CH_6_OX_B431_WE=int.from_bytes(bytes[43:47], byteorder='little', signed=False)
            ADC_CH_6_OX_B431_WE=(ADC_CH_6_OX_B431_WE*2.5)/(2**31)
            ADC_CH_7_OX_B431_AUX=int.from_bytes(bytes[47:51], byteorder='little', signed=False)
            ADC_CH_7_OX_B431_AUX=(ADC_CH_7_OX_B431_AUX*2.5)/(2**31)
            TLB=int.from_bytes(bytes[51:52], byteorder='little', signed=False)
            THB=int.from_bytes(bytes[52:53], byteorder='little', signed=False)
            Outside_Temperature = ((THB*64+TLB/4)/2**14)*165-40
            RLB=int.from_bytes(bytes[53:54], byteorder='little', signed=False)
            RHB=int.from_bytes(bytes[54:55], byteorder='little', signed=False)
            Outside_Humidity=((RHB*256+RLB)/2**14)*100
            Voltage_VCC_main=int.from_bytes(bytes[55:57], byteorder='little', signed=False)
            Voltage_VCC_main=(Voltage_VCC_main/2**12)*(22750/2750)
            Voltage_5V_total=int.from_bytes(bytes[57:59], byteorder='little', signed=False)
            Voltage_5V_total=(Voltage_5V_total/2**12)*(15200/3100)
            Voltage_5V_ISB=int.from_bytes(bytes[59:61], byteorder='little', signed=False)
            Voltage_5V_ISB=(Voltage_5V_ISB/2**12)*(15200/3100)
            Voltage_4V=int.from_bytes(bytes[61:63], byteorder='little', signed=False)
            Voltage_4V=(Voltage_4V/2**12)*(20100/5100)
            Voltage_3_3_V=int.from_bytes(bytes[63:65], byteorder='little', signed=False)
            Voltage_3_3_V=(Voltage_3_3_V/2**12)*(15750/4750)
            Serialnumber_uc1=int.from_bytes(bytes[65:69], byteorder='little', signed=False)
            Serialnumber_uc2=int.from_bytes(bytes[69:73], byteorder='little', signed=False)
            Serialnumber_uc3=int.from_bytes(bytes[73:77], byteorder='little', signed=False)
            Serialnumber_uc4=int.from_bytes(bytes[77:81], byteorder='little', signed=False)
    #       dt=[year,month,day,hour,minute,second,pressure,temperature,humidity,ADC_CH_0_CO_B4_WE,
    #          ADC_CH_1_CO_B4_AUX,ADC_CH_2_NO_B4_WE,ADC_CH_3_NO_B4_AUX,ADC_CH_4_NO2_B43F_WE,
    #          ADC_CH_5_NO2_B43F_AUX,ADC_CH_6_OX_B431_WE,ADC_CH_7_OX_B431_AUX,TLB,THB,Outside_Temperature,RLB,RHB,
    #          Outside_Humidity,Voltage_VCC_main,Voltage_5V_total,Voltage_5V_ISB,Voltage_4V,Voltage_3_3_V,Serialnumber_uc1,
    #          Serialnumber_uc2,Serialnumber_uc3,Serialnumber_uc4]
            dt=[year,month,day,hour,minute,second,ADC_CH_0_CO_B4_WE]
            data.append(dt)
    counter1+=1
    print(counter1)

#df= pd.DataFrame(data, columns=["year","month","day","hour","minute","second","pressure","temperature","humidity",
#                                "ADC_CH_0_CO_B4_WE","ADC_CH_1_CO_B4_AUX",
#                                "ADC_CH_2_NO_B4_WE","ADC_CH_3_NO_B4_AUX","ADC_CH_4_NO2_B43F_WE","ADC_CH_5_NO2_B43F_AUX",
#                                "ADC_CH_6_OX_B431_WE","ADC_CH_7_OX_B431_AUX","TLB","THB","Outside_Temperature","RLB","RHB","Outside_Humidity","Voltage_VCC_main",
#                                "Voltage_5V_total","Voltage_5V_ISB","Voltage_4V","Voltage_3_3_V","Serialnumber_uc1",
#                                "Serialnumber_uc2","Serialnumber_uc3","Serialnumber_uc4"])
df= pd.DataFrame(data, columns=['year','month','day','hour','minute','second','CO'])
#df.to_csv('Alphasense_output.csv', index=False)

directory2 = r"C:\Users\MARTA\Documents\Alphasense\2020-03-12\C24B4(10tebervlidan 12 martamde)"
dt2=[]
data2=[]
counter2 = 0
for entry in os.scandir(directory2):
    if(counter2==12):
        break
    with open(entry,"rb") as f:
        while (bytes := f.read(81)):
            year=int.from_bytes(bytes[0:2], byteorder='little', signed=False)
            month=int.from_bytes(bytes[2:3], byteorder='little', signed=False)         
            day=int.from_bytes(bytes[3:4], byteorder='little', signed=False)
            hour=int.from_bytes(bytes[4:5], byteorder='little', signed=False)
            minute=int.from_bytes(bytes[5:6], byteorder='little', signed=False)
            second=int.from_bytes(bytes[6:7], byteorder='little', signed=False)
            pressure=int.from_bytes(bytes[7:11], byteorder='little', signed=False)
            temperature=int.from_bytes(bytes[11:15], byteorder='little', signed=False)
            humidity=int.from_bytes(bytes[15:19], byteorder='little', signed=False)
            ADC_CH_0_CO_B4_WE=int.from_bytes(bytes[19:23], byteorder='little', signed=False)
            ADC_CH_0_CO_B4_WE=(ADC_CH_0_CO_B4_WE*2.5)/(2**31)
            ADC_CH_1_CO_B4_AUX=int.from_bytes(bytes[23:27], byteorder='little', signed=False)
            ADC_CH_1_CO_B4_AUX=(ADC_CH_1_CO_B4_AUX*2.5)/(2**31)
            ADC_CH_2_NO_B4_WE=int.from_bytes(bytes[27:31], byteorder='little', signed=False)
            ADC_CH_2_NO_B4_WE=(ADC_CH_2_NO_B4_WE*2.5)/(2**31)
            ADC_CH_3_NO_B4_AUX=int.from_bytes(bytes[31:35], byteorder='little', signed=False)
            ADC_CH_3_NO_B4_AUX=(ADC_CH_3_NO_B4_AUX*2.5)/(2**31)
            ADC_CH_4_NO2_B43F_WE=int.from_bytes(bytes[35:39], byteorder='little', signed=False)
            ADC_CH_4_NO2_B43F_WE=(ADC_CH_4_NO2_B43F_WE*2.5)/(2**31)
            ADC_CH_5_NO2_B43F_AUX=int.from_bytes(bytes[39:43], byteorder='little', signed=False)
            ADC_CH_5_NO2_B43F_AUX=(ADC_CH_5_NO2_B43F_AUX*2.5)/(2**31)
            ADC_CH_6_OX_B431_WE=int.from_bytes(bytes[43:47], byteorder='little', signed=False)
            ADC_CH_6_OX_B431_WE=(ADC_CH_6_OX_B431_WE*2.5)/(2**31)
            ADC_CH_7_OX_B431_AUX=int.from_bytes(bytes[47:51], byteorder='little', signed=False)
            ADC_CH_7_OX_B431_AUX=(ADC_CH_7_OX_B431_AUX*2.5)/(2**31)
            TLB=int.from_bytes(bytes[51:52], byteorder='little', signed=False)
            THB=int.from_bytes(bytes[52:53], byteorder='little', signed=False)
            Outside_Temperature = ((THB*64+TLB/4)/2**14)*165-40
            RLB=int.from_bytes(bytes[53:54], byteorder='little', signed=False)
            RHB=int.from_bytes(bytes[54:55], byteorder='little', signed=False)
            Outside_Humidity=((RHB*256+RLB)/2**14)*100
            Voltage_VCC_main=int.from_bytes(bytes[55:57], byteorder='little', signed=False)
            Voltage_VCC_main=(Voltage_VCC_main/2**12)*(22750/2750)
            Voltage_5V_total=int.from_bytes(bytes[57:59], byteorder='little', signed=False)
            Voltage_5V_total=(Voltage_5V_total/2**12)*(15200/3100)
            Voltage_5V_ISB=int.from_bytes(bytes[59:61], byteorder='little', signed=False)
            Voltage_5V_ISB=(Voltage_5V_ISB/2**12)*(15200/3100)
            Voltage_4V=int.from_bytes(bytes[61:63], byteorder='little', signed=False)
            Voltage_4V=(Voltage_4V/2**12)*(20100/5100)
            Voltage_3_3_V=int.from_bytes(bytes[63:65], byteorder='little', signed=False)
            Voltage_3_3_V=(Voltage_3_3_V/2**12)*(15750/4750)
            Serialnumber_uc1=int.from_bytes(bytes[65:69], byteorder='little', signed=False)
            Serialnumber_uc2=int.from_bytes(bytes[69:73], byteorder='little', signed=False)
            Serialnumber_uc3=int.from_bytes(bytes[73:77], byteorder='little', signed=False)
            Serialnumber_uc4=int.from_bytes(bytes[77:81], byteorder='little', signed=False)
    #       dt=[year,month,day,hour,minute,second,pressure,temperature,humidity,ADC_CH_0_CO_B4_WE,
    #          ADC_CH_1_CO_B4_AUX,ADC_CH_2_NO_B4_WE,ADC_CH_3_NO_B4_AUX,ADC_CH_4_NO2_B43F_WE,
    #          ADC_CH_5_NO2_B43F_AUX,ADC_CH_6_OX_B431_WE,ADC_CH_7_OX_B431_AUX,TLB,THB,Outside_Temperature,RLB,RHB,
    #          Outside_Humidity,Voltage_VCC_main,Voltage_5V_total,Voltage_5V_ISB,Voltage_4V,Voltage_3_3_V,Serialnumber_uc1,
    #          Serialnumber_uc2,Serialnumber_uc3,Serialnumber_uc4]
            dt2=[year,month,day,hour,minute,second,ADC_CH_0_CO_B4_WE]
            data2.append(dt2)
    counter2+=1
    print(counter2)
df2 = pd.DataFrame(data2, columns=['year','month','day','hour','minute','second','CO'])

df_inner1 = pd.merge(df,df2,on=["year","month","day","hour","minute","second" ],how="inner")

#df_inner1.to_csv('Alphasense_output.csv', index=False)
r'''
plt.plot(df['CO'],'b.')
plt.xlabel('time')
plt.ylabel('CO_B2')
plt.show()
plt.plot(df2['CO'],'b.')
plt.xlabel('time')
plt.ylabel('CO_B4')
plt.show()
'''

#pikaro
df_pikaro = pd.DataFrame()
directory3 = r'C:\Users\MARTA\Documents\pikaro\02'
counter3 = 0
#first for cycle goes through each directory in main directory
for entry1 in os.scandir(directory3):
    if(counter3==10):
        break
    #second for cycle goes through each file in subdirectory
    for entry2 in os.scandir(entry1):
        #this line reads opens the file and reads each line
        datContent = [i.strip().split() for i in open(entry2.path).readlines()]
        #this line creates a dataframe
        df_temp = DataFrame (datContent[1:] ,columns=datContent[0])
        #this line appends the dataframe from this file to the previous file
        df_pikaro = df_pikaro.append(df_temp, ignore_index = True)
    counter3+=1
    print(counter3)



df_array=df_pikaro.to_numpy()


df_height = df_pikaro.iloc[:,1].size

data = []
i = 0
while(i<df_height):
    year = int(df_array[i][0][:4])
    month = int(df_array[i][0][5:7])
    day = int(df_array[i][0][8:10])
    hour = int(df_array[i][1][:2])
    minute = int(df_array[i][1][3:5])
    second = int(df_array[i][1][6:8])
    CO = float(df_array[i][17][:-4])/10
    CO2 = float(df_array[i][18])
    CO2_dry = float(df_array[i][19])
    CH4 = float(df_array[i][20])
    CH4_dry = float(df_array[i][21])
    H2O = float(df_array[i][22])
    arr = [year,month,day,hour,minute,second,CO,CO2,CO2_dry,CH4,CH4_dry,H2O]
    data.append(arr)
    i+=1
    print(i)

df2_pikaro = pd.DataFrame(data, columns=["year","month","day", "hour", "minute", "second", "CO_Picaro",
                                         "CO2","CO2_dry","CH4","CH4_dry","H2O"])


#pikaros da alphasensis gaertianeba
df_final = df2_pikaro.groupby(['year','month','day','hour','minute',
                               'second'])['CO_Picaro','CO2','CO2_dry','CH4','CH4_dry','H2O'].mean()

df_final.to_csv('pikaro.csv')
df_inner2 = pd.merge(df_inner1,df_final,on=["year","month","day","hour","minute","second" ],how="inner")



r'''
boards_average = np.add(df_inner2.iloc[:,6],df_inner2.iloc[:,7])/2
df_inner2['Boards_average'] = boards_average

diff_pikaro_boards = np.subtract(df_inner2.iloc[:,8], boards_average)
df_inner2['diff_pikaro_boards'] = diff_pikaro_boards

diff_B2_pikaro=np.subtract(df_inner2.iloc[:,6],df_inner2.iloc[:,8])
diff_B4_pikaro=np.subtract(df_inner2.iloc[:,7],df_inner2.iloc[:,8])



B2_error_1st_norm=np.linalg.norm(diff_B2_pikaro, ord=1)/np.linalg.norm(df_inner2.iloc[:,8], ord=1)

B4_error_1st_norm=np.linalg.norm(diff_B4_pikaro, ord=1)/np.linalg.norm(df_inner2.iloc[:,8], ord=1)

boards_average_error_1st_norm=np.linalg.norm(diff_pikaro_boards, ord=1)/np.linalg.norm(df_inner2.iloc[:,8], ord=1)


B2_error_2nd_norm=np.linalg.norm(diff_B2_pikaro, ord=2)/np.linalg.norm(df_inner2.iloc[:,8], ord=2)

B4_error_2nd_norm=np.linalg.norm(diff_B4_pikaro, ord=2)/np.linalg.norm(df_inner2.iloc[:,8], ord=2)

boards_average_error_2nd_norm=np.linalg.norm(diff_pikaro_boards, ord=2)/np.linalg.norm(df_inner2.iloc[:,8], ord=2)


B2_error_oo_norm=abs(diff_B2_pikaro).max()/(abs(df_inner2.iloc[:,8]).max())

B4_error_oo_norm=abs(diff_B4_pikaro).max()/(abs(df_inner2.iloc[:,8]).max())

boards_average_error_oo_norm=abs(diff_pikaro_boards).max()/(abs(df_inner2.iloc[:,8]).max())

norm_data = [[B2_error_1st_norm,B4_error_1st_norm,boards_average_error_1st_norm,B2_error_2nd_norm,B4_error_2nd_norm,
             boards_average_error_2nd_norm,B2_error_oo_norm,B4_error_oo_norm,boards_average_error_oo_norm]]
df_norms=pd.DataFrame(norm_data, columns=["B2_error_1st_norm","B4_error_1st_norm","boards_aveage_error_1st_norm",
                                          "B2_error_2nd_norm","B4_error_2nd_norm","boards_average_error_2nd_norm",
                                          "B2_error_oo_norm","B4_error_oo_norm","boards_average_error_oo_norm"])

print('last')
#df_inner2.to_csv('mixed.csv')
writer = pd.ExcelWriter('./output.xlsx', engine='xlsxwriter')
df_inner2.to_excel(writer, sheet_name='Sheet1')
df_norms.to_excel(writer, sheet_name='Sheet2')
writer.save()

'''
r'''

#wutobrivi sashualo
CO_B2 = df_inner2.iloc[:,6]
CO_B4 = df_inner2.iloc[:,7]
CO_Picaro = df_inner2.iloc[:,8]


CO_B2_mnt_avg = []
CO_B4_mnt_avg = []
CO_Picaro_mnt_avg = []

i=0
left=CO_Picaro.size

B2_first_norm = []
B4_first_norm = []
BA_first_norm = []
B2_second_norm = []
B4_second_norm = []
BA_second_norm = []
B2_oo_norm = []
B4_oo_norm = []
BA_oo_norm = []

while(left>0):
    if(left>60):        
        j=i+60
        s1=sum(CO_B2[i:j])/60
        CO_B2_mnt_avg=np.append(CO_B2_mnt_avg,s1)
        s2=sum(CO_B4[i:j])/60
        CO_B4_mnt_avg=np.append(CO_B4_mnt_avg,s2)
        s3=sum(CO_Picaro[i:j])/60
        CO_Picaro_mnt_avg=np.append(CO_Picaro_mnt_avg,s3)
        
        boards_average_mnt = np.add(CO_B2[i:j],CO_B4[i:j])/2
        diff_picaro_boards_mnt = np.subtract(CO_Picaro[i:j],boards_average_mnt)
        diff_B2_picaro_mnt = np.subtract(CO_B2[i:j],CO_Picaro[i:j])
        diff_B4_picaro_mnt = np.subtract(CO_B4[i:j],CO_Picaro[i:j])
        
        B2_error_1st_norm_mnt=np.linalg.norm(diff_B2_picaro_mnt, ord=1)/np.linalg.norm(CO_Picaro[i:j], ord=1)
        B2_first_norm.append(B2_error_1st_norm_mnt)
        B4_error_1st_norm_mnt=np.linalg.norm(diff_B4_picaro_mnt, ord=1)/np.linalg.norm(CO_Picaro[i:j], ord=1)  
        B4_first_norm.append(B4_error_1st_norm_mnt)
        boards_average_error_1st_norm_mnt=np.linalg.norm(diff_picaro_boards_mnt, ord=1)/np.linalg.norm(CO_Picaro[i:j], ord=1)
        BA_first_norm.append(boards_average_error_1st_norm_mnt)

        B2_error_2nd_norm_mnt=np.linalg.norm(diff_B2_picaro_mnt, ord=2)/np.linalg.norm(CO_Picaro[i:j], ord=2)
        B2_second_norm.append(B2_error_2nd_norm_mnt)
        B4_error_2nd_norm_mnt=np.linalg.norm(diff_B4_picaro_mnt, ord=2)/np.linalg.norm(CO_Picaro[i:j], ord=2)
        B4_second_norm.append(B4_error_2nd_norm_mnt)
        boards_average_error_2nd_norm_mnt=np.linalg.norm(diff_picaro_boards_mnt, ord=2)/np.linalg.norm(CO_Picaro[i:j], ord=2)
        BA_second_norm.append(boards_average_error_2nd_norm_mnt)
        
        B2_error_oo_norm_mnt=abs(diff_B2_picaro_mnt).max()/(abs(CO_Picaro[i:j])).max()
        B2_oo_norm.append(B2_error_oo_norm_mnt)
        B4_error_oo_norm_mnt=(abs(diff_B4_picaro_mnt)).max()/(abs(CO_Picaro[i:j])).max()
        B4_oo_norm.append(B4_error_oo_norm_mnt)
        boards_average_error_oo_norm_mnt=(abs(diff_picaro_boards_mnt)).max()/(abs(CO_Picaro[i:j])).max()
        BA_oo_norm.append(boards_average_error_oo_norm_mnt)
        
    else:
        j=i+left
        s1=sum(CO_B2[i:j])/left
        CO_B2_mnt_avg=np.append(CO_B2_mnt_avg,s1)
        s2=sum(CO_B4[i:j])/left
        CO_B4_mnt_avg=np.append(CO_B4_mnt_avg,s2)
        s3=sum(CO_Picaro[i:j])/left
        CO_Picaro_mnt_avg=np.append(CO_Picaro_mnt_avg,s3)
        
        boards_average_mnt = np.add(CO_B2[i:j],CO_B4[i:j])/2
        diff_picaro_boards_mnt = np.subtract(CO_Picaro[i:j],boards_average_mnt)
        diff_B2_picaro_mnt = np.subtract(CO_B2[i:j],CO_Picaro[i:j])
        diff_B4_picaro_mnt = np.subtract(CO_B4[i:j],CO_Picaro[i:j])
        
        boards_average_mnt = np.add(CO_B2[i:j],CO_B4[i:j])/2
        diff_picaro_boards_mnt = np.subtract(CO_Picaro[i:j],boards_average_mnt)
        diff_B2_picaro_mnt = np.subtract(CO_B2[i:j],CO_Picaro[i:j])
        diff_B4_picaro_mnt = np.subtract(CO_B4[i:j],CO_Picaro[i:j])
        
        B2_error_1st_norm_mnt=np.linalg.norm(diff_B2_picaro_mnt, ord=1)/np.linalg.norm(CO_Picaro[i:j], ord=1)
        B2_first_norm.append(B2_error_1st_norm_mnt)
        B4_error_1st_norm_mnt=np.linalg.norm(diff_B4_picaro_mnt, ord=1)/np.linalg.norm(CO_Picaro[i:j], ord=1)  
        B4_first_norm.append(B4_error_1st_norm_mnt)
        boards_average_error_1st_norm_mnt=np.linalg.norm(diff_picaro_boards_mnt, ord=1)/np.linalg.norm(CO_Picaro[i:j], ord=1)
        BA_first_norm.append(boards_average_error_1st_norm_mnt)

        B2_error_2nd_norm_mnt=np.linalg.norm(diff_B2_picaro_mnt, ord=2)/np.linalg.norm(CO_Picaro[i:j], ord=2)
        B2_second_norm.append(B2_error_2nd_norm_mnt)
        B4_error_2nd_norm_mnt=np.linalg.norm(diff_B4_picaro_mnt, ord=2)/np.linalg.norm(CO_Picaro[i:j], ord=2)
        B4_second_norm.append(B4_error_2nd_norm_mnt)
        boards_average_error_2nd_norm_mnt=np.linalg.norm(diff_picaro_boards_mnt, ord=2)/np.linalg.norm(CO_Picaro[i:j], ord=2)
        BA_second_norm.append(boards_average_error_2nd_norm_mnt)
        
        B2_error_oo_norm_mnt=abs(diff_B2_picaro_mnt).max()/abs(CO_Picaro[i:j]).max()
        B2_oo_norm.append(B2_error_oo_norm_mnt)
        B4_error_oo_norm_mnt=abs(diff_B4_picaro_mnt).max()/abs(CO_Picaro[i:j]).max()
        B4_oo_norm.append(B4_error_oo_norm_mnt)
        boards_average_error_oo_norm_mnt=abs(diff_picaro_boards_mnt).max()/abs(CO_Picaro[i:j]).max()
        BA_oo_norm.append(boards_average_error_oo_norm_mnt)
        
    i+=60
    left = left-60

#print(CO_B2_mnt_avg,CO_B4_mnt_avg,CO_Picaro_mnt_avg)

df_mnt = pd.DataFrame()
df_mnt['CO_B2_mnt_avg']=CO_B2_mnt_avg
df_mnt['CO_B4_mnt_avg']=CO_B4_mnt_avg
df_mnt['CO_Picaro_mnt_avg']=CO_Picaro_mnt_avg
df_mnt['B2_first_norm']=B2_first_norm
df_mnt['B4_first_norm']=B4_first_norm
df_mnt['Boards_average_first_norm']=BA_first_norm
df_mnt['B2_second_norm']=B2_second_norm
df_mnt['B4_second_norm']=B4_second_norm
df_mnt['Boards_average_second_norm']=BA_second_norm
df_mnt['B2_oo_norm']=B2_oo_norm
df_mnt['B4_oo_norm']=B4_oo_norm
df_mnt['Boards_average_oo_norm']=BA_oo_norm
#print(df_mnt)
df_mnt.to_csv('wutobrivi.csv')

'''








  
r"""
#wutobrivis sashualos svetebis sashualo
df_average3 = np.add(CO_mnt_avg,NO_mnt_avg)/2
df_average4 = np.add(NO2_mnt_avg,OX_mnt_avg)/2
df_average_mnt = np.add(df_average3,df_average4)/2
#print(df_average_mnt)


#sensorebis shedarebebi mat sashualostan-wamobrivi
#CO-s shedareba sashualostan-wamobrivi
CO_difference=np.subtract(df.iloc[:,10],df_average)
#NO-s shedareba sashualostan-wamobrivi
NO_difference=np.subtract(df.iloc[:,12],df_average)
#NO2-s shedareba sashualostan-wamobrivi
NO2_difference=np.subtract(df.iloc[:,14],df_average)
#OX-is shedareba sashualostan-wamobrivi
OX_difference=np.subtract(df.iloc[:,16],df_average)
#print(CO_difference,NO_difference,NO2_difference,OX_difference)

#sensorebis shedarebebi mat sashualostan-wutobrivi
#CO-s shedareba sashualostan-wutobrivi
CO_difference_mnt=np.subtract(CO_mnt_avg,df_average_mnt)
#NO-s shedareba sashualostan-wutobrivi
NO_difference_mnt=np.subtract(NO_mnt_avg,df_average_mnt)
#NO2-s shedareba sashualostan-wutobrivi
NO2_difference_mnt=np.subtract(NO2_mnt_avg,df_average_mnt)
#OX-is shedareba sashualostan-wutobrivi
OX_difference_mnt=np.subtract(OX_mnt_avg,df_average_mnt)
#print(CO_difference_mnt,NO_difference_mnt,NO2_difference_mnt,OX_difference_mnt)


#1-li veqtoruli normit cdomilebis datvla-wamobrivistvis
#CO-stvis
CO_1st_norm=np.linalg.norm(CO_difference, ord=1)/np.linalg.norm(df_average, ord=1)
#NO-stvis
NO_1st_norm=np.linalg.norm(NO_difference, ord=1)/np.linalg.norm(df_average, ord=1)
#NO2-stvis
NO2_1st_norm=np.linalg.norm(NO2_difference, ord=1)/np.linalg.norm(df_average, ord=1)
#OX-stvis
OX_1st_norm=np.linalg.norm(OX_difference, ord=1)/np.linalg.norm(df_average, ord=1)
print(CO_1st_norm,NO_1st_norm,NO2_1st_norm,OX_1st_norm)

#me-2 veqtoruli normit cdomilebis datvla-wamobrivistvis
#CO-stvis
CO_2nd_norm=np.linalg.norm(CO_difference, ord=2)/np.linalg.norm(df_average, ord=2)
#NO-stvis
NO_2nd_norm=np.linalg.norm(NO_difference, ord=2)/np.linalg.norm(df_average, ord=2)
#NO2-stvis
NO2_2nd_norm=np.linalg.norm(NO2_difference, ord=2)/np.linalg.norm(df_average, ord=2)
#OX-stvis
OX_2nd_norm=np.linalg.norm(OX_difference, ord=2)/np.linalg.norm(df_average, ord=2)
#print(CO_2nd_norm,NO_2nd_norm,NO2_2nd_norm,OX_2nd_norm)


#oo veqtoruli normit cdomilebis datvla-wamobrivistvis
#CO-stvis
CO_oo_norm=abs(CO_difference).max()/abs(df_average).max()
#NO-stvis
NO_oo_norm=abs(NO_difference).max()/abs(df_average).max()
#NO2-stvis
NO2_oo_norm=abs(NO2_difference).max()/abs(df_average).max()
#OX-stvis
OX_oo_norm=abs(OX_difference).max()/abs(df_average).max()
#print(CO_oo_norm,NO_oo_norm,NO2_oo_norm,OX_oo_norm)

#1-li veqtoruli normit cdomilebis datvla-wutobrivistvis
#CO-stvis
CO_1st_norm_mnt=np.linalg.norm(CO_difference_mnt, ord=1)/np.linalg.norm(df_average_mnt, ord=1)
#NO-stvis
NO_1st_norm_mnt=np.linalg.norm(NO_difference_mnt, ord=1)/np.linalg.norm(df_average_mnt, ord=1)
#NO2-stvis
NO2_1st_norm_mnt=np.linalg.norm(NO2_difference_mnt, ord=1)/np.linalg.norm(df_average_mnt, ord=1)
#OX-stvis
OX_1st_norm_mnt=np.linalg.norm(OX_difference_mnt, ord=1)/np.linalg.norm(df_average_mnt, ord=1)
#print(CO_1st_norm_mnt,NO_1st_norm_mnt,NO2_1st_norm_mnt,OX_1st_norm_mnt)

#me-2 veqtoruli normit cdomilebis datvla-wutobrivistvis
#CO-stvis
CO_2nd_norm_mnt=np.linalg.norm(CO_difference_mnt, ord=2)/np.linalg.norm(df_average_mnt, ord=2)
#NO-stvis
NO_2nd_norm_mnt=np.linalg.norm(NO_difference_mnt, ord=2)/np.linalg.norm(df_average_mnt, ord=2)
#NO2-stvis
NO2_2nd_norm_mnt=np.linalg.norm(NO2_difference_mnt, ord=2)/np.linalg.norm(df_average_mnt, ord=2)
#OX-stvis
OX_2nd_norm_mnt=np.linalg.norm(OX_difference_mnt, ord=2)/np.linalg.norm(df_average_mnt, ord=2)
#print(CO_1st_norm_mnt,NO_1st_norm_mnt,NO2_1st_norm_mnt,OX_1st_norm_mnt)

#oo veqtoruli normit cdomilebis datvla-wutobrivistvis
#CO-stvis
CO_oo_norm_mnt=abs(CO_difference_mnt).max()/abs(df_average_mnt).max()
#NO-stvis
NO_oo_norm_mnt=abs(NO_difference_mnt).max()/abs(df_average_mnt).max()
#NO2-stvis
NO2_oo_norm_mnt=abs(NO2_difference_mnt).max()/abs(df_average_mnt).max()
#OX-stvis
OX_oo_norm_mnt=abs(OX_difference_mnt).max()/abs(df_average_mnt).max()
#print(CO_oo_norm_mnt,NO_oo_norm_mnt,NO2_oo_norm_mnt,OX_oo_norm_mnt)



j=0
dt_sec = []
empty_array = []
while(j<df_average.size):
    dt_sec=[df_average[j],CO_difference[j],NO_difference[j],NO2_difference[j],OX_difference[j]]
    empty_array.append(dt_sec)
    j+=1
df_sec = pd.DataFrame(empty_array,  columns=["df_average","CO_difference","NO_difference","NO2_difference","OX_difference"])
#df_sec.to_csv('Alphasense_output2.csv', index=False)

x=0
dt_mnt = []
empty_array2 = []
while(x<df_average_mnt.size):
    dt_mnt=[df_average_mnt[x],CO_difference_mnt[x],NO_difference_mnt[x],NO2_difference_mnt[x],OX_difference_mnt[x]]
    empty_array2.append(dt_mnt)
    x+=1
df_mnt = pd.DataFrame(empty_array2,  columns=["df_average_mnt","CO_difference_mnt","NO_difference_mnt","NO2_difference_mnt","OX_difference_mnt"])
#df_mnt.to_csv('Alphasense_output2.csv', index=False)               
writer = pd.ExcelWriter('./blabal.xlsx', engine='xlsxwriter')
df.to_excel(writer, sheet_name='Sheet1')
df1.to_excel(writer, sheet_name='Sheet2')
writer.save()
"""











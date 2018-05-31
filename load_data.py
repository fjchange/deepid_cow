import csv
import numpy as np
import pickle
from PIL import Image
import  random

basic_path="/home/shikigan/res/03270849/"

def vectorize_imgs(img_path):
    with Image.open(img_path) as img:
        #对图像进行变换压缩，太大了，而且不规则
        img=img.resize((47,55))
        arr_img = np.asarray(img, dtype='float32')
        return arr_img

def csv_dict(path):
    dict_content=dict()
    with open(path,'r')as c:
        reader=csv.reader(c)
        for row in reader:
            row[1]=row[1].replace("\'","")
            if row[2]==-1:
                continue
            if(dict_content.has_key(row[2])):
                dict_content[row[2]].append(row[1]+'.jpg')
            else:
                dict_content[row[2]]=[row[1]]
    return dict_content

def common_csv_list(path):
    X_list = list()
    Y_list = list()
    with open(path, 'r')as c:
        reader = csv.reader(c)
        for row in reader:
            X_list.append(row[0])
            row[1]=int(row[1])
            Y_list.append([row[1]])
    return X_list, Y_list

def csv_list(path):
    X_list = list()
    Y_list=list()
    with open(path, 'r')as c:
        reader = csv.reader(c)
        for row in reader:
            row[1] = row[1].replace("\'", "")
            row[3]=int(row[3])
            if row[3] == -1:
                continue
            X_list.append(basic_path+row[1]+'/'+str(row[2])+'.jpg')
            row[3]=int(row[3])
            Y_list.append([row[3]])
    return X_list,Y_list


def load_data():
    with open('data/dataset.pkl', 'rb') as f:
        testX1=pickle.load(f)
        testX2=pickle.load(f)
        testY=pickle.load(f)
        validX=pickle.load(f)
        validY=pickle.load(f)
        trainX = pickle.load(f)
        trainY = pickle.load(f)
        return testX1,testX2, testY, validX,validY,trainX, trainY

def read_csv_file(path):
    x_list,y_list=common_csv_list(path)
    x=[]
    for item in x_list:
        x.append(vectorize_imgs(item))
    return x,y_list

def create_csv_pair_file(path,test_percent):
    x_list,y_list=csv_list(path)
    test_size=int(len(x_list)*test_percent)
    count=0
    with open('pair_test_set.csv','w')as c:
        writer=csv.writer(c,delimiter=',')
        while(count<test_size):
            idx1=int(random.random()*len(x_list))
            idx2=idx1+1
            if idx2>=len(x_list)-1:
                idx2=idx1-1
            if y_list[idx1]!=-1 and y_list[idx2]!=-1:
                temp_ans=0
                if y_list[idx1]==y_list[idx2]:
                    temp_ans=1
                count+=1
                writer.writerow([x_list[idx1],x_list[idx2],temp_ans])
        print("create success")

def creat_train_valid_file(path,valid_percent):
    #待后续改善，使验证集与训练集分离。

    x_list,y_list=csv_list(path)
    valid_size=int(len(x_list)*valid_percent)
    li=list(range(len(x_list)))
    index_np=np.array(li)
    np.random.shuffle(index_np)

    count=0
    with open('valid_set.csv','w')as c:
        writer=csv.writer(c,delimiter=',')
        while(count<valid_size):
            if(y_list[index_np[count]]!=-1):
                writer.writerow([x_list[index_np[count]],y_list[index_np[count]][0]])
            count+=1
    with open('train_set.csv','w')as c:
        writer=csv.writer(c,delimiter=',')
        while(count<len(x_list)):
            if(y_list[index_np[count]]!=-1):
                writer.writerow([x_list[index_np[count]],y_list[index_np[count]][0]])
            count+=1
        print("create success")

def read_csv_pair_file(path):
    x_list1,x_list2,y_list=[],[],[]
    with open(path,'r')as c:
        reader=csv.reader(c)
        for row in reader:
            x_list1.append(vectorize_imgs(row[0]))
            x_list2.append(vectorize_imgs(row[1]))
            row[2]=int(row[2])
            y_list.append([row[2]])
    return x_list1,x_list2,y_list

if __name__ == '__main__':
    create_csv_pair_file("/home/shikigan/kiwi_fung/labeled_data/03270849.csv",0.2)
    creat_train_valid_file("/home/shikigan/kiwi_fung/labeled_data/03270849.csv",0.1)


    testX1, testX2, testY = read_csv_pair_file('pair_test_set.csv')
    validX,validY=read_csv_file('valid_set.csv')
    trainX, trainY = read_csv_file("train_set.csv")

    print(testX1.__len__(), testX2.__len__(), testY.__len__())
    print(trainX.__len__(), trainY.__len__())

    with open('data/dataset.pkl', 'wb') as f:
        pickle.dump(testX1, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(testX2, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(testY , f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(validX, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(validY, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(trainX, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(trainY, f, pickle.HIGHEST_PROTOCOL)



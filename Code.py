#IMPORTANT, for the code to run as fast as possible use the module cupy, but the module numpy also
#suffices to a certain degree.
import cupy as np
import json
import re
import random


#The function that drowns the irrelevnt parts of the txt

#!!!To run any code, you first need to create a Net with chosen parametrs, change the parameters to your liking on line 350.!!!


def init(name):
    with open("stop_words_english.txt", encoding='utf-8') as f:
        stop_words = f.read().splitlines()
        stop_words = set(stop_words)
    with open(name, encoding='utf-8') as f:
        training = f.read()
    training = training.lower()
    training = re.sub(r"[^a-zA-Z\s]+", "", training)
    tr_list = training.split()
    for word in tr_list:
        if word in stop_words:
            tr_list.remove(word)
    training = " ".join(tr_list)
    training_list = training.split()
    with open("init.json", 'w') as json_file:
        json.dump(training_list, json_file)
    print("The text has been prepared for further work.")

#This creates the consectuive list of words in the text

def consec():
     with open("init.json", encoding='utf-8') as f:
        training = json.load(f)
     temp = []
     for i in training:
          if i not in temp:
               temp.append(i)
     with open("consec.json", 'w') as json_file:
         json.dump(temp, json_file)
     print("The list of consequtive first apperances has been created.")

#A function that transforms the text into numbers

def transform():
    with open('init.json', encoding='utf-8') as f:
        corpus = json.load(f)
        with open('vec', encoding='utf-8') as t:
            dict = json.load(t)
            temp=[]
            for i in corpus: 
                temp.append(dict[i])
    with open("corpus.json", 'w') as json_file:
        json.dump(temp, json_file)
    print("The text has been translated.")
        

# this is a function that pinpoints the number of vectors we are going to need to describe a text 

def number_words():
    with open("init.json", encoding='utf-8') as f:
                pokus = json.load(f)
    data2_set = set(pokus)
    return print(int(len(pokus)), int(len(data2_set)))

# This is a function that takes kern as a central text that we work with and n, which is how many context words do we want for each definition

def Data(n: int):
    d = dict()
    with open("corpus.json", encoding='utf-8') as f:
        kern = json.load(f)
    for word in range(0, len(kern)):
        if kern[word] in d:
            d[kern[word]].append((kern[word+1: word+n+1]))
        else:
            d[kern[word]]=[]
            d[kern[word]].append(kern[word+1: word+n+1])
    file_path = 'Data.json'
    with open(file_path,"w") as json_file:
        json.dump(d, json_file)
    print("The data for each word has been stored.")

# A simple fucntion to count the instances od words in your text. Usefull to predict if your
# words have enough instances to be embeddded, the industry minimum is in the tens, but for a
# comprehensive dictionary you need at least a hundred of instances.

def Word_Count():
    with open('init.json', encoding='utf-8') as f:
        corpus = json.load(f)
    dict = {}
    for i in corpus:
        if i in dict:
            dict[i] = dict[i]+1
        else:
             dict[i] = 1
    with open("dict","w") as json_file:
        json.dump(dict, json_file)  

# Now we need to encode the words as vectors 

def vec():
    with open('consec.json', encoding='utf-8') as f:
                pokus = json.load(f)
    word_dict ={word: i for  i, word in enumerate(pokus)}
    with open("vec","w") as json_file:
        json.dump(word_dict, json_file)

#The reverse of the last function

def ivec():
    with open('vec', encoding='utf-8') as f:
                pokus = json.load(f)
    inverted_dict = {v: k for k, v in pokus.items()}
    with open("ivec","w") as json_file:
        json.dump(inverted_dict, json_file)

#The function that splits the training data and test data for you

def split(l):
    with open('Data.json', encoding='utf-8') as f:
        temp = json.load(f)
        result = [(int(key), value) for key in temp for value in temp[key]]
        k = len(result)
        random.shuffle(result)
        split = int(l * k)
        train_data = result[:split]
        test_data = result[split:]
        with open('train_data.json',"w") as json_file:
            json.dump(train_data, json_file)
        with open('test_data.json',"w") as json_file:
            json.dump(test_data, json_file)
    print("The data has been split correctly.")

#This function removes all the words below the instances limit k 

def drown(k, name):
    if k == 0:
         return
    with open(name, encoding='utf-8') as f:
        corpus = json.load(f)
    with open('dict', encoding='utf-8') as f:
        word_counts = json.load(f)
    corpus = [i for i in corpus if word_counts.get(i, 0) >= k]

    with open(name, "w") as json_file:
        json.dump(corpus, json_file)

#The function that makes this usable, the name is the name of the text you want to transform
#This function creates seperate files to work with, init.json is the function stripped of all redundancies, consec is the list of consequtive words that appered in the file.
#Data.json is the actutal file with training data with its corrsponding numbers and training and testint lists are the lists to train and test your model on, you can choose
#the percentage of train and test data with k, which should be value bounded by 0 and 1 where 1 is everything and zero is nothing 
#The last file this will create is the vec and ivec files, where vec is the dictionary from numbers to words and ivec is the dictionary for words to numbers

def Prepare_Text(name, k, drown_n=0):
     try:
        init(name)
     except FileNotFoundError:
        print("Error: Ensure that the name is correct and in a correct format")
     except json.JSONDecodeError:
        print("Error: There was an issue decoding the file.")
     except Exception as e:
        print("An unexpected error occurred")
     try:
          drown(drown_n)
     except TypeError:
        print("Error: One of the arguments is missing or is in the wrong format.")
     consec()
     vec()
     ivec()
     transform()
     try:
        Data("corpus.json",5)
     except FileNotFoundError:
        print("Error: Ensure that the name is correct and in a correct format")
     except json.JSONDecodeError:
        print("Error: There was an issue decoding the file.")
     except TypeError:
        print("Error: One of the arguments is missing or is in the wrong format.")
     try:
          split(k)
     except TypeError:
        print("Error: The argument k is missing or is in the wrong format.")
     drown(drown_n)
     print("hotovo")

# The model itself

try:
    with open('corpus.json', encoding='utf-8') as f:
                KORPUS = json.load(f)
                n = max(KORPUS)
                print("Neural network is prepered to be run.")
except FileNotFoundError:
    print("To run functions concerning the model, first prepare the text for learning.")


#These are the non linear functions

def ReLu(x):
    return np.maximum(x,0)

#The softmax function is the probability function used as the last layer

def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)

#Non linear function sigmoid used in neural networks

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#One hot encoding thhe vector using the variable num

def one_hot_encode(num):
    return np.eye(n+1)[num]

#derivatives

# derivative of sigmoid

def sigmoidPrime(X):
        s = sigmoid(X)
        return np.multiply(s ,1-s)

#derivative of ReLu

def dev_Relu(x):
    return x>0

#The function to normalize a vector

def normalize_vector(v):
    v = np.array(v)
    return np.divide(v, np.linalg.norm(v))

#Derivative of a function that orthognalizes vectors

def dev_orthogonal_regularization(matrix):
        G = np.dot(matrix, matrix.T) 
        identity = np.eye(matrix.shape[0])
        gradient = 4 * 1e-4 * np.dot(matrix, (G - identity))
        return gradient


class Net:

    def __init__(self,first,second,alpha):
        self.alpha = alpha
        self.linear1  =  np.random.normal(0, 2/np.sqrt(5*(n+1)),size=(first, 5*(n+1)))
        self.bias1 = np.zeros(first,)
        self.linear2 = np.random.normal(0,1/np.sqrt(first),size=(second,first))
        self.bias2  = np.zeros(second,)
        self.linear3 = np.random.normal(0, 1/np.sqrt(second), size = (n+1, second))
        self.bias3 = np.zeros(n+1,)
        self.gW1 = np.zeros_like(self.linear1)
        self.gB1 = np.zeros_like(self.bias1)
        self.gW2 = np.zeros_like(self.linear2)
        self.gB2 = np.zeros_like(self.bias2)
        self.gW3 = np.zeros_like(self.linear3)
        self.gB3 = np.zeros_like(self.bias3)
    
    def forward(self,x):
        Z1 = np.matmul(self.linear1,x) + self.bias1
        A1 = ReLu(Z1)
        Z2 = np.matmul(self.linear2,A1) + self.bias2
        A2 = sigmoid(Z2)
        Z3 = np.matmul(self.linear3,A2) + self.bias3
        A3 = softmax(Z3) 
        return Z1, A1, Z2, A2, Z3, A3
    
#one cycle of backpropagation

    def cycle(self, x, Y):
        Z1, A1, Z2, A2, Z3, A3 = self.forward(x)
        dZ3 = A3-Y
        dB3 = dZ3
        dW3 = np.dot(dZ3[:,None],A2[None,:])
        dZ2 = np.dot(np.transpose(self.linear3),dZ3[:,None]) * dev_Relu(Z2)[:,None]
        dB2 = np.concatenate(dZ2)
        dW2 = np.dot(dZ2,A1[None,:])
        dZ1 = np.dot(np.transpose(self.linear2),dZ2)*sigmoidPrime(Z1)[:,None]
        dB1 = np.concatenate(dZ1)
        dW1 = dZ1*x[None,:]
        dW1 += dev_orthogonal_regularization(self.linear1)
        dW2 += dev_orthogonal_regularization(self.linear2)
        dW3 += dev_orthogonal_regularization(self.linear3)

        return  dW1, dB1, dW2, dB2, dW3, dB3
    
    def correct(self, dW1, dB1, dW2, dB2, dW3, dB3, Arda = True):
        epsilon = 1e-8

        self.gW1 += dW1 ** 2
        self.gB1 += dB1 ** 2
        self.gW2 += dW2 ** 2
        self.gB2 += dB2 ** 2
        self.gW3 += dW3 ** 2
        self.gB3 += dB3 ** 2
        if Arda == True:
            self.linear1 -= self.alpha / (np.sqrt(self.gW1) + epsilon) * dW1
            self.bias1 -= self.alpha / (np.sqrt(self.gB1) + epsilon) * dB1
            self.linear2 -= self.alpha / (np.sqrt(self.gW2) + epsilon) * dW2
            self.bias2 -= self.alpha / (np.sqrt(self.gB2) + epsilon) * dB2
            self.linear3 -= self.alpha / (np.sqrt(self.gW3) + epsilon) * dW3
            self.bias3 -= self.alpha / (np.sqrt(self.gB3) + epsilon) * dB3
        else:
            self.linear1 -= self.alpha * dW1
            self.bias1 -= self.alpha * dB1
            self.linear2 -= self.alpha * dW2
            self.bias2 -= self.alpha * dB2
            self.linear3 -= self.alpha * dW3
            self.bias3 -= self.alpha * dB3

        return print("corrected")

    #This saves the neural network for future purposes

    def save(self,name):
        linear1 = self.linear1
        linear2 = self.linear2
        linear3 = self.linear3
        bias1 = self.bias1
        bias2 = self.bias2
        bias3 = self.bias3   
        linear1c, bias1c, linear2c, bias2c, linear3c, bias3c = np.asnumpy(linear1), np.asnumpy(bias1), np.asnumpy(linear2), np.asnumpy(bias2), np.asnumpy(linear3), np.asnumpy(bias3)
        linear1, bias1, linear2, bias2, linear3, bias3 = linear1c.tolist(), bias1c.tolist(), linear2c.tolist(), bias2c.tolist(), linear3c.tolist(), bias3c.tolist()
        temp = [linear1, bias1, linear2, bias2, linear3, bias3]
        with open(name,"w") as json_file:
            json.dump(temp, json_file)
        return linear1, bias1, linear2, bias2, linear3, bias3
    
    #This function loades a previosly saved neural network

    def load(self,name):
        with open(name, encoding='utf-8') as f:
            temp = json.load(f)
        linear1, bias1, linear2, bias2, linear3, bias3 = temp
        linear1, bias1, linear2, bias2, linear3, bias3 = np.array(linear1), np.array(bias1), np.array(linear2), np.array(bias2), np.array(linear3), np.array(bias3)
        self.linear1 = linear1
        self.linear2 = linear2
        self.linear3 = linear3
        self.bias1 = bias1
        self.bias2 = bias2
        self.bias3 = bias3
        return print("hotovo")

#This is the actuall function that backpropagates the network, the batch size is the number of vectors of gradient descent that you want to avarage over and epoch is the
#number of batches you want to run, it is recommended to have batches lowet than 20 for strong computers and less than weaker desktops. Non CUDA machines are limited by their
#RAM, so bath_sizes dont play such a big role
#"name" is the name of the file that will save your file after training



try:
    net = Net(400,300, 0.01)
except NameError:
    x=0

def Backprop(batch_size,epoch,name, Autograd = True):
    with open('train_data.json', encoding='utf-8') as f:
                training_list = json.load(f)
    if len(training_list)  < batch_size *epoch:
         return print("The training data is not long enough for desired forwarding, please prepare a longer text or change the trainging requirements")
    for i in range(0,epoch):
        temp = []
        for j in range(0,batch_size):
            temp.append(list(net.cycle(np.concatenate(one_hot_encode(training_list[j][1])),one_hot_encode(training_list[j][0]))))
        correction = []
        for k in range(0,len(temp[0])):
            temp2 =[]
            for h in temp:
                temp2.append(h[k])
            temp3 = np.array(temp2)
            result = np.mean(temp3, axis=0)
            correction.append(result)
        dW1, dB1, dW2, dB2, dW3, dB3 = correction
        net.correct(dW1, dB1, dW2, dB2, dW3, dB3, Autograd) 
        print(i)
    net.save(name)

#Takes the name of the net and the name of the translation library, name is the name of the neural netwrok, name2 is the translation to words and name3 is the name that saves your file

def translate(name,name2,name3):
    net.load(name)
    Weight = net.linear3
    for i in range(0,len(Weight)):
         Weight[i] =  np.divide(Weight[i],np.linalg.norm(Weight[i]))
    Weight2 = np.asnumpy(Weight)
    Weight2 = Weight2.tolist()
    with open(name2, encoding='utf-8') as f:
                pokus = json.load(f)
    temp ={}
    for i in range(0,len(pokus)):
        temp[pokus[str(i)]] = Weight2[i]       
    with open(name3, 'w') as json_file:
        json.dump(temp, json_file)

#This function compares the values of embedding vectors, it intakes two embedding vectors of the same size

def cosine_similarity(first, second):
    first = np.array(first)
    second = np.array(second)
    return np.dot(first,second)

#This function takes words, translates them and theen compares them using the file where you saved your embedding vectors, 1 means that the vectors are the same and -1 means that
#the vectors are completely different

def cosine_similarity_word(first, second, Embedding_vector):
    with open(Embedding_vector, encoding='utf-8') as f:
                pokus = json.load(f) 
    first_vector = pokus[first]
    second_vector = pokus[second]
    return print(cosine_similarity(first_vector,second_vector))

#This takes a word and prints the list o n most similliar words

def similliar(word,n, Embedding_vector):
    with open(Embedding_vector, encoding='utf-8') as f:
                pokus = json.load(f) 
    similarities = {}
    word_vector = pokus[word]
    for other_word, other_vector in pokus.items():
        similarities[other_word] = cosine_similarity(word_vector, other_vector)
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return print(sorted_similarities[:n])

def least_similliar(word,n, Embedding_vector):
    with open(Embedding_vector, encoding='utf-8') as f:
                pokus = json.load(f) 
    similarities = {}
    word_vector = pokus[word]
    for other_word, other_vector in pokus.items():
        similarities[other_word] = cosine_similarity(word_vector, other_vector)
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return print(sorted_similarities[n:])

#This searches the file with the name "name" using a file named Embedding_vector for a "word"

def search(word,name, Embedding_vector):
    with open(name, encoding='utf-8') as f:
                text = json.load(f)  
    with open(Embedding_vector, encoding='utf-8') as f:
                EV = json.load(f) 
    temp = []
    for i in text:
         x = cosine_similarity(word,EV[str(i)])
         temp.append((x,i))
    temp.sort()
    print(temp[:n])

#The actuall training function

def training_function(Batch_size, epochs, name,k):
    try:
        drown(k)
    except TypeError:
        print("Error: The argument k is missing or is in the wrong format.")
    try:
        Backprop(batch_size=Batch_size,epoch=epochs,name=name)
    except FileNotFoundError:
        print("Error: Ensure that the name is correct and in a correct format")
    except json.JSONDecodeError:
        print("Error: There was an issue decoding the file.")
    except TypeError:
        print("Error: One of the arguments is missing or is in the wrong format.")
    translate(name,"ivec",name + "Embdedding")
    print("The embedding dictionary has been saved under your chosen name with the suffix embedding.")

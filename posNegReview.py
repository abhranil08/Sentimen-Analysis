import pandas as pd
df = pd.read_csv("zomato.csv")
df.head()
df1 = df[['name','reviews_list']]
df1.head()

#reviews = df['reviews_list']
#a = reviews[0]

def posReviewCol(a):
    pos, neg = 0,0
    a=a[1:].split()
    for i in range(1,len(a)):
        if a[i-1]=="('Rated":
            b=float(a[i][:-2])
            if b>=3.5:
                pos+=1
    return pos

def negReviewCol(a):
    pos, neg = 0,0
    a=a[1:].split()
    for i in range(1,len(a)):
        if a[i-1]=="('Rated":
            b=float(a[i][:-2])
            if b<=3.0:
                neg+=1
    return neg

df1["PositiveReviews"] = df1['reviews_list'].apply(lambda x: posReviewCol(x))
df1["NegativeReviews"] = df1['reviews_list'].apply(lambda x: negReviewCol(x))


    
    
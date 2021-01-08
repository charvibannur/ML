# import librairies
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math as math
import time
plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = [14,14]

# load the data
df = pd.read_csv(r"C:\Users\charv\Desktop\Charvi\movies_2.csv")
# convert to datetime
df["date_added"] = pd.to_datetime(df['date_added'])
df['year'] = df['date_added'].dt.year
df['month'] = df['date_added'].dt.month
df['day'] = df['date_added'].dt.day
# convert columns "director, listed_in, production and country" in columns that contain a real list
# the strip function is applied on the elements
# if the value is NaN, the new column contains a empty list []
df['categories'] = df['listed_in'].apply(lambda l: [] if pd.isna(l) else [i.strip() for i in l.split(",")])
df['production'] = df['production_companies'].apply(lambda l: [] if pd.isna(l) else [i.strip() for i in l.split(",")])
df['countries'] = df['production_countries'].apply(lambda l: [] if pd.isna(l) else [i.strip() for i in l.split(",")])

df.head()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.cluster import MiniBatchKMeans

# Build the tfidf matrix with the descriptions
start_time = time.time()
text_content = df['description']
vector = TfidfVectorizer(  # only use words that appear at least X times
    stop_words='english',  # remove stop words
    lowercase=True,  # Convert everything to lower case
    use_idf=True,  # Use idf
    norm=u'l2',  # Normalization
    smooth_idf=True  # Prevents divide-by-zero errors
)
df.description = df.description.fillna(' ')
tfidf = vector.fit_transform(text_content)

# Clustering  Kmeans
k = 200
kmeans = MiniBatchKMeans(n_clusters=k)
kmeans.fit(tfidf)
centers = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = vector.get_feature_names()

# print the centers of the clusters
# for i in range(0,k):
#     word_list=[]
#     print("cluster%d:"% i)
#     for j in centers[i,:10]:
#         word_list.append(terms[j])
#     print(word_list)

request_transform = vector.transform(df['description'])
# new column cluster based on the description
df['cluster'] = kmeans.predict(request_transform)

# Find similar : get the top_n movies with description similar to the target description
def find_similar(tfidf_matrix, index, top_n = 5):
    cosine_similarities = linear_kernel(tfidf_matrix[index:index+1], tfidf_matrix).flatten()
    related_docs_indices = [i for i in cosine_similarities.argsort()[::-1] if i != index]
    return [index for index in related_docs_indices][0:top_n]


G = nx.Graph(label="MOVIE")
start_time = time.time()
for i, rowi in df.iterrows():
    if (i % 1000 == 0):
        print(" iter {} -- {} seconds --".format(i, time.time() - start_time))
    G.add_node(rowi['title'], key=rowi['show_id'], label="MOVIE", mtype=rowi['type'], rating=rowi['rating'])
    #    G.add_node(rowi['cluster'],label="CLUSTER")
    #    G.add_edge(rowi['title'], rowi['cluster'], label="DESCRIPTION")
    for element in rowi['production_companies']:
        G.add_node(element, label="PERSON")
        G.add_edge(rowi['title'], element, label="PRODUCTION_IN")
    for element in rowi['categories']:
        G.add_node(element, label="CAT")
        G.add_edge(rowi['title'], element, label="CAT_IN")

    for element in rowi['countries']:
        G.add_node(element, label="COU")
        G.add_edge(rowi['title'], element, label="COU_IN")

    indices = find_similar(tfidf, i, top_n=5)
    snode = "Sim(" + rowi['title'][:15].strip() + ")"
    G.add_node(snode, label="SIMILAR")
    G.add_edge(rowi['title'], snode, label="SIMILARITY")
    for element in indices:
        G.add_edge(snode, df['title'].loc[element], label="SIMILARITY")
print(" finish -- {} seconds --".format(time.time() - start_time))

def get_all_adj_nodes(list_in):
    sub_graph=set()
    for m in list_in:
        sub_graph.add(m)
        for e in G.neighbors(m):
                sub_graph.add(e)
    return list(sub_graph)
def draw_sub_graph(sub_graph):
    subgraph = G.subgraph(sub_graph)
    colors=[]
    for e in subgraph.nodes():
        if G.nodes[e]['label']=="MOVIE":
            colors.append('blue')
        elif G.nodes[e]['label']=="PERSON":
            colors.append('red')
        elif G.nodes[e]['label']=="CAT":
            colors.append('green')
        elif G.nodes[e]['label']=="COU":
            colors.append('yellow')
        elif G.nodes[e]['label']=="SIMILAR":
            colors.append('orange')
        elif G.nodes[e]['label']=="CLUSTER":
            colors.append('orange')

    nx.draw(subgraph, with_labels=True, font_weight='bold',node_color=colors)
    plt.show()

list_in = ["Ninja Assassin", "Die Hard 2"]
sub_graph = get_all_adj_nodes(list_in)
draw_sub_graph(sub_graph)


def get_recommendation(root):
    commons_dict = {}
    for e in G.neighbors(root):
        for e2 in G.neighbors(e):
            if e2 == root:
                continue
            if G.nodes[e2]['label'] == "MOVIE":
                commons = commons_dict.get(e2)
                if commons == None:
                    commons_dict.update({e2: [e]})
                else:
                    commons.append(e)
                    commons_dict.update({e2: commons})
    movies = []
    weight = []
    for key, values in commons_dict.items():
        w = 0.0
        for e in values:
            w = w + 1 / math.log(G.degree(e))
        movies.append(key)
        weight.append(w)

    result = pd.Series(data=np.array(weight), index=movies)
    result.sort_values(inplace=True, ascending=False)
    return result;
#gui
from tkinter import *

root = Tk()

root.title("NETFLIX RECOMMENDATION SYSTEM")
root.geometry("666x450")

C = Canvas(root, height=150, width=200)
filename = PhotoImage(file = "C:\\Users\\charv\\PycharmProjects\\pythonProject\\netflix.png")
background_label = Label(root, image=filename)
background_label.place(x=0, y=0, relwidth=1, relheight=1)
C.pack()

def run_recommendation():
    input_value = retrieve_input()
    result = get_recommendation(input_value)
    print("*" * 40 + "\n Recommendation for " + input_value + "\n" + "*" * 40)
    print(result.head())
    textbox2.insert(END, result.head())

def retrieve_input():
    return textbox.get("1.0","end-1c")


textbox= Text(root, height=1, width=10 ,borderwidth=1 )
textbox.place(x=300, y=40)

mybutton = Button(root, text="Get Recommendations",command=lambda: run_recommendation())
mybutton.place(x=285,y=80)

textbox2= Text(root, height=7, width=50 ,borderwidth=1 )
textbox2.place(x=120, y=300)

root.resizable(False, False)

root.mainloop()
#end of gui



from django.shortcuts import render

import os
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from django.http import HttpResponse, JsonResponse
import matplotlib.pyplot as plt
from pylab import *
from matplotlib import pylab
import PIL, PIL.Image
from io import BytesIO
import base64
import pandas as pd
import numpy as np
import chart_studio
import plotly.graph_objects as go
import chart_studio.plotly as py
import chart_studio.tools as tls
import plotly.graph_objs as pgo
import plotly.express as px

import glob

load_dotenv()

matplotlib.use('Agg')

username = "3nws"
API_KEY = os.getenv('PLOTLY_API_KEY')

tls.set_credentials_file(username=username, api_key=API_KEY)

def normalize(matrix):
    norm = np.linalg.norm(matrix)
    matrix = matrix/norm  # normalized matrix
    return matrix

# Create your views here.

def analysis(request):
    return render(request, './home.html')

def k_means(request):
    return render(request, './k-means.html')
 
def result(request):
    dataset = pd.read_csv(request.FILES['csv_file'],header=0)
    # will create a form model later
    form = request.POST
    column_1 = int(form['column_1'])
    column_2 = int(form['column_2'])
    num_of_clusters = int(form['cluster-count'])
    max_iterations = int(form['max_iter'])
    kmeans_algo = str(form['kmeans_algo'])
    
    X=dataset.iloc[:, [column_1,column_2]].values
    fig = px.scatter(x=dataset.iloc[:, column_1].values,y=dataset.iloc[:, column_2].values)
    py.plot(fig, filename= "k-means-scatter", auto_open=False)
    # fig.write_html('static/plots/first_figure.html', auto_open=True) I LL TRY THIS LATER
    
    try:
        if form['isNormalized']:
            X = normalize(X)
    except KeyError:
        pass
    
    # unclustered
    plt.scatter(X[:, 0], X[:, 1])
    plt.title('Unclustered data')
    plt.xlabel(dataset.columns[column_1])
    plt.ylabel(dataset.columns[column_2])
    grid(True)
    plt.savefig("static/img/unclustered.png", transparent=True)
    
    # Store image in a buffer
    buffer_unclustered = BytesIO()
    canvas = pylab.get_current_fig_manager().canvas
    canvas.draw()
    pilImage = PIL.Image.frombytes("RGB", canvas.get_width_height(), canvas.tostring_rgb())
    pilImage.save(buffer_unclustered, "PNG")
    pylab.close()
    
    unclustered = (base64.b64encode(buffer_unclustered.getvalue())).decode('utf8')
    
    
    kmeans = KMeans(n_clusters=num_of_clusters, init ='k-means++', max_iter=300, n_init=10,random_state=0, algorithm=kmeans_algo)
    y_kmeans = kmeans.fit_predict(X)
    plt.scatter(X[y_kmeans==0, 0], X[y_kmeans==0, 1], s=100, c='red', label ='Cluster 1')
    plt.scatter(X[y_kmeans==1, 0], X[y_kmeans==1, 1], s=100, c='blue', label ='Cluster 2')
    plt.scatter(X[y_kmeans==2, 0], X[y_kmeans==2, 1], s=100, c='green', label ='Cluster 3')
    plt.scatter(X[y_kmeans==3, 0], X[y_kmeans==3, 1], s=100, c='cyan', label ='Cluster 4')
    plt.scatter(X[y_kmeans==4, 0], X[y_kmeans==4, 1], s=100, c='magenta', label ='Cluster 5')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label = 'Centroids')
    plt.title('Clusters')
    plt.xlabel(dataset.columns[column_1])
    plt.ylabel(dataset.columns[column_2])
    grid(True)
    plt.savefig("static/img/clusters.png", transparent=True)
    
    
 
    # Store image in a buffer
    buffer1 = BytesIO()
    canvas = pylab.get_current_fig_manager().canvas
    canvas.draw()
    pilImage = PIL.Image.frombytes("RGB", canvas.get_width_height(), canvas.tostring_rgb())
    pilImage.save(buffer1, "PNG")
    pylab.close()
 
    # elbow
    ssd = []
    for i in range(1, 10):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 5)
        kmeans.fit(X)
        ssd.append(kmeans.inertia_)
    plt.plot(range(1, 10), ssd)
    plt.grid()
    plt.title('The Elbow Method', fontsize =10)
    plt.xlabel('Number of clusters', fontsize =10)
    plt.ylabel('SSD',fontsize =10)
    grid(True)
    plt.savefig("static/img/elbow.png", transparent=True)
    
    result = (base64.b64encode(buffer1.getvalue())).decode('utf8')
 
    # Store image in a buffer
    buffer2 = BytesIO()
    canvas = pylab.get_current_fig_manager().canvas
    canvas.draw()
    pilImage = PIL.Image.frombytes("RGB", canvas.get_width_height(), canvas.tostring_rgb())
    pilImage.save(buffer2, "PNG")
    pylab.close()
 
    elbow = (base64.b64encode(buffer2.getvalue())).decode('utf8')
    
    context = {
        'result': result,
        'elbow': elbow,
        'unclustered': unclustered,
    }
    return render(request, './result.html', context=context)
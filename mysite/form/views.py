from django.shortcuts import render

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

# Create your views here.

def index(request):
    return render(request, './index.html')
 
def result(request):
    dataset = pd.read_csv(request.FILES['csv_file'])
    form = request.POST
    column_1 = int(form['column_1'])
    column_2 = int(form['column_2'])
    num_of_clusters = int(form['cluster-count'])
    max_iterations = int(form['max_iter'])
    X=dataset.iloc[:, [column_1,column_2]].values
    kmeans = KMeans(n_clusters=num_of_clusters, init ='k-means++', max_iter=300, n_init=10,random_state=0 )
    y_kmeans = kmeans.fit_predict(X)
    plt.scatter(X[y_kmeans==0, 0], X[y_kmeans==0, 1], s=100, c='red', label ='Cluster 1')
    plt.scatter(X[y_kmeans==1, 0], X[y_kmeans==1, 1], s=100, c='blue', label ='Cluster 2')
    plt.scatter(X[y_kmeans==2, 0], X[y_kmeans==2, 1], s=100, c='green', label ='Cluster 3')
    plt.scatter(X[y_kmeans==3, 0], X[y_kmeans==3, 1], s=100, c='cyan', label ='Cluster 4')
    plt.scatter(X[y_kmeans==4, 0], X[y_kmeans==4, 1], s=100, c='magenta', label ='Cluster 5')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label = 'Centroids')
    plt.title('Clusters')
    plt.xlabel('x')
    plt.ylabel('y')
    grid(True) # grey lines my man
 
    # Store image in a buffer
    buffer = BytesIO()
    canvas = pylab.get_current_fig_manager().canvas
    canvas.draw()
    pilImage = PIL.Image.frombytes("RGB", canvas.get_width_height(), canvas.tostring_rgb())
    pilImage.save(buffer, "PNG")
    pylab.close()
 
    # Send buffer in a http response the the browser with the mime type image/png set
    return HttpResponse(buffer.getvalue(), content_type="image/png")
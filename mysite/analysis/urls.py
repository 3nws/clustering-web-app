from django.urls import path

from . import views

urlpatterns = [
    path('result', views.result, name='result'),
    path('k-means', views.k_means, name='k-means'),
    path('', views.analysis, name='analysis'),
]
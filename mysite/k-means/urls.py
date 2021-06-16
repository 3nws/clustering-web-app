from django.urls import path

from . import views

urlpatterns = [
    path('result', views.result, name='result'),
    path('', views.k_means, name='k-means'),
]
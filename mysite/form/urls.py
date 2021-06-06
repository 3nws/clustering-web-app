from django.urls import path

from . import views

urlpatterns = [
    path('showimage', views.showimage, name='showimage'),
    path('result', views.upload_csv, name='upload_csv'),
    path('', views.index, name='index'),
]
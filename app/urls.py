from django.contrib import admin
from django.urls import path,include
from . import views

urlpatterns=[
    path("",views.index,name='index'),
    path("about/",views.about,name='about'),
    path("login/",views.login,name='login'),
    path("registration/",views.registration,name='registration'),
    path("userhome/",views.userhome,name='userhome'),
    path("load/",views.load,name='load'),
    path("view/",views.view,name='view'),
    path("preprocessing/",views.preprocessing,name='preprocessing'),
    path("model/",views.model,name='model'),
    path("prediction/",views.prediction,name='prediction'),
    # path("graph/",views.graph,name='graph'),
    
]
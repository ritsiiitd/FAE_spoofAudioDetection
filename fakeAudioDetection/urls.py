from django.contrib import admin
from django.urls import path
from fakeAudioDetection import views
urlpatterns = [
    path("",views.index,name='home'),
    path("play",views.play,name='play'),
    path("home",views.index,name='home'),
]


from django.urls import path
from . import views

urlpatterns = [
    path('', views.Classification.as_view(), name='home')
]

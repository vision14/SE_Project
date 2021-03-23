from django.urls import path
from . import views

urlpatterns = [
    path('', views.Classification.as_view(), name='admin_home'),
    path('regression', views.Regression.as_view(), name='regression')
]

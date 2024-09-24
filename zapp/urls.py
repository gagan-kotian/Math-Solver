from django.urls import path
from . import views

urlpatterns = [
    path('linear_regression/', views.linear_regression, name='linear_regression'),
    path('numerical_integration/', views.numerical_integration, name='numerical_integration'),
    path('', views.demo, name='demo'),
    path('interpolation_polynomial/', views.interpolation_polynomial, name='interpolation_polynomial'),
    path('interpolation_unequal/', views.interpolation_unequal, name='interpolation_unequal'),
    path('newtongregory/', views.newtongregory, name='newtongregory'),
    path('numerical_differentiation/', views.numerical_differentiation, name='numerical_differentiation'),
    path('curve_fitting/', views.curve_fitting, name='curve_fitting'),
]

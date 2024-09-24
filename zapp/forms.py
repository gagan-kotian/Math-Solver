from django import forms

class AdditionForm(forms.Form):
    number1 = forms.CharField(label='Number 1')
    number2 = forms.CharField(label='Number 2')
class NumericalIntegrationForm(forms.Form):
    step = forms.IntegerField(label='step')
    yvalues = forms.CharField(label='yvalues')
class InterpolationForm(forms.Form):
    xvalue = forms.CharField(label='xvalue')
    yvalue = forms.CharField(label='yvalue')
class UnequalForm(forms.Form):
    xvalues = forms.CharField(label='xvalues')
    yvalues = forms.CharField(label='yvalues')
    poi = forms.FloatField(label='poi')



from django import forms

class MyForm(forms.Form):
    api_url = forms.CharField(widget=forms.HiddenInput())
    my_field = forms.CharField()
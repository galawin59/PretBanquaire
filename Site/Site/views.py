from django.shortcuts import render
from .forms import *

import requests

def home_page(request):
    return render(request,"site/homePage.html")


# def home_page(request):
#     if request.method == 'POST':
#         form = MyForm(request.POST)
#         if form.is_valid():
#             data = form.cleaned_data
#             api_url = data['api_url']
#             my_field = data['my_field']
#             payload = {'my_field': my_field}
#             response = requests.post(api_url, data=payload)
#             # Utilisez la réponse pour afficher les résultats
#     else:
#         form = MyForm()
#     return render(request, 'site/homePage.html', {'form': form})
from django.shortcuts import render
from .forms import InputForm
import pickle
from django.conf import settings
import os

# Create your views here.
with open(os.path.join(settings.MEDIA_ROOT, 'my_classifier.pkl'), 'rb') as file:
    classifier = pickle.load(file)


def home_view(request):

    message = ""
    age = 0
    salary = 0
    submitbutton = request.POST.get("submit")

    form = InputForm(request.POST or None)
    if form.is_valid():
        age = int(form.cleaned_data.get("age"))
        salary = int(form.cleaned_data.get("salary"))

        preds = classifier.predict([[age, salary]])

        if preds == 0:
            message = "Customer will not buy the product"
        else:
            message = "Customer will buy the product"

    context = {'form': form, 'message': message, 'submitbutton': submitbutton}

    return render(request, 'user_classifier/base.html', context)

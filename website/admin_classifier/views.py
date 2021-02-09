from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import os
import pickle
from django.conf import settings
import pandas as pd
from io import StringIO
from .forms import InputForm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

data = pd.DataFrame({})

# Create your views here.


def upload(request):
    global data
    n_neighbors = 5
    leaf_size = 30
    weights = 'uniform'
    algorithm = 'auto'

    context = {}
    submitbutton = request.POST.get("submit")
    form = InputForm(request.POST or None)

    if request.method == 'POST':
        if 'pkl' in request.FILES:
            upload_file = request.FILES['pkl']
            if upload_file.content_type == 'application/octet-stream':
                fs = FileSystemStorage()
                os.remove(os.path.join(settings.MEDIA_ROOT, 'my_classifier.pkl'))
                fs.save('my_classifier.pkl', upload_file)
                context['pkl_message'] = "File Uploaded"
            else:
                context['pkl_message'] = "Invalid File Type"
        elif 'csv' in request.FILES:
            upload_file = request.FILES['csv']
            if upload_file.content_type == 'application/vnd.ms-excel':
                string_data = StringIO(upload_file.read().decode('utf-8'))
                data = pd.read_csv(string_data)
                context['csv_message'] = "File Uploaded"
            else:
                context['csv_message'] = "Invalid File Type"
    if request.method == 'POST' and form.is_valid():
        n_neighbors = int(form.cleaned_data.get("n_neighbors"))
        leaf_size = int(form.cleaned_data.get("leaf_size"))
        weights = form.cleaned_data.get("weights")
        algorithm = form.cleaned_data.get("algorithm")

        X = data.iloc[:, [2, 3]].values
        y = data.iloc[:, -1].values

        classifier = KNeighborsClassifier(n_neighbors=n_neighbors,
                                          leaf_size=leaf_size,
                                          weights=weights,
                                          algorithm=algorithm)
        classifier.fit(X, y)
        y_pred = classifier.predict(X)

        os.remove(os.path.join(settings.MEDIA_ROOT, 'my_classifier.pkl'))
        with open(os.path.join(settings.MEDIA_ROOT, 'my_classifier.pkl'), 'wb') as file:
            pickle.dump(classifier, file)

        accuracy = accuracy_score(y, y_pred)

        context['message'] = "Model Successfully Trained"
        context['submitbutton'] = submitbutton
        context['accuracy'] = round(accuracy*100, 2)

    context['form'] = form
    context['submitbutton'] = submitbutton

    return render(request, 'admin_classifier/admin_base.html', context)


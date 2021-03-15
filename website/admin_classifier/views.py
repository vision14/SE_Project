from django.shortcuts import render
import pickle
import pandas as pd
from io import StringIO
from .forms import InputForm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from django.views import View
from pymongo import MongoClient
from bson.binary import Binary

data = pd.DataFrame({})

# Create your views here.


class Algorithm(View):

    def get(self, request):
        pass

    def post(self, request):
        pass


class Classification(Algorithm):

    form_class = InputForm
    template_name = 'admin_classifier/admin_base.html'
    submit_button = None
    message = None
    pkl_message = None
    csv_message = None
    accuracy = None

    def get(self, request):

        form = self.form_class()
        context = {'form': form, 'submitbutton': self.submit_button}

        return render(request, self.template_name, context)

    def post(self, request):

        global data

        form = self.form_class(request.POST)
        self.submit_button = request.POST.get("submit")

        if 'pkl' in request.FILES:
            upload_file = request.FILES['pkl']
            if upload_file.content_type == 'application/octet-stream':
                pkl_obj = pickle.dumps(upload_file)
                mongo_data = {'pkl_data': Binary(pkl_obj)}
                try:
                    client = MongoClient(
                        "mongodb+srv://user_1:USER_1@cluster0.0oqke.mongodb.net/<dbname>?retryWrites=true&w=majority")
                    db = client.get_database('learnml_db')
                    db_data = db.classification
                    db_data.delete_many({})
                    db_data.insert_one(mongo_data)
                    self.pkl_message = "File Uploaded"
                except:
                    self.pkl_message = "Unexpected error while uploading pickle file"
            else:
                self.pkl_message = "Invalid File Type"
        elif 'csv' in request.FILES:
            upload_file = request.FILES['csv']
            if upload_file.content_type == 'application/vnd.ms-excel':
                string_data = StringIO(upload_file.read().decode('utf-8'))
                data = pd.read_csv(string_data)
                self.csv_message = "File Uploaded"
            else:
                self.csv_message = "Invalid File Type"

        if form.is_valid():
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

            pkl_obj = pickle.dumps(classifier)
            mongo_data = {'pkl_data': Binary(pkl_obj)}
            try:
                client = MongoClient(
                    "mongodb+srv://user_1:USER_1@cluster0.0oqke.mongodb.net/<dbname>?retryWrites=true&w=majority")
                db = client.get_database('learnml_db')
                db_data = db.classification
                db_data.delete_many({})
                db_data.insert_one(mongo_data)
                self.accuracy = round(accuracy_score(y, y_pred)*100, 2)
                self.message = "Model Successfully Trained"
            except:
                self.message = "Unexpected error while training the model."

        context = {'form': form, 'submitbutton': self.submit_button, 'pkl_message': self.pkl_message,
                   'csv_message': self.csv_message, 'accuracy': self.accuracy, 'message': self.message}

        return render(request, self.template_name, context)

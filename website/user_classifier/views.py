from django.shortcuts import render
from .forms import InputForm
import pickle
from django.conf import settings
import os
from django.views import View
from pymongo import MongoClient

# Create your views here.
client = MongoClient("mongodb+srv://user_1:USER_1@cluster0.0oqke.mongodb.net/<dbname>?retryWrites=true&w=majority")
db = client.get_database('learnml_db')
db_data = db.classification
data = list(db_data.find())[0]['pkl_data']
classifier = pickle.loads(data)


class Algorithm(View):

    def get(self, request):
        pass

    def post(self, request):
        pass


class Classification(Algorithm):

    form_class = InputForm
    template_name = 'user_classifier/base.html'
    message = ""
    submit_button = None

    def get(self, request):

        form = self.form_class()
        context = {'form': form, 'message': self.message, 'submitbutton': self.submit_button}

        return render(request, self.template_name, context)

    def post(self, request):

        form = self.form_class(request.POST)
        self.submit_button = request.POST.get("submit")

        if form.is_valid():
            age = int(form.cleaned_data.get("age"))
            salary = int(form.cleaned_data.get("salary"))
            preds = classifier.predict([[age, salary]])

            if preds == 0:
                self.message = "No, the customer will not buy the product"
            else:
                self.message = "Yes, the customer will buy the product"

        context = {'form': form, 'message': self.message, 'submitbutton': self.submit_button}

        return render(request, self.template_name, context)

from django.shortcuts import render
import pickle
from django.views import View
from pymongo import MongoClient
import numpy as np

# Create your views here.
client = MongoClient("mongodb+srv://user_1:USER_1@cluster0.0oqke.mongodb.net/<dbname>?retryWrites=true&w=majority")
db = client.get_database('learnml_db')
db_data = db.algorithms


class Algorithm(View):

    def get(self, request):
        pass

    def post(self, request):
        pass


class Classification(Algorithm):

    template_name = 'user_classifier/generic.html'
    message = ""
    submit_button = None

    def get(self, request):

        data = db_data.find_one({'name': 'KNN'})
        context = {'algo_desc': data['algo_desc'], 'ds_desc': data['ds_desc'],
                   'training_features': data['training_features']}

        return render(request, self.template_name, context)

    def post(self, request):

        data = db_data.find_one({'name': 'KNN'})
        graph_image = data['graph_image']
        if data['upload_method'] == 'pkl':
            classifier = pickle.loads(pickle.loads(data['pkl_data']).read())
        else:
            classifier = pickle.loads(data['pkl_data'])
        if 'submit' in request.POST:
            self.submit_button = request.POST.get("submit")
            output_message = data['training_label']
            user_inputs = np.array(request.POST.getlist('user_inputs')).astype(np.float64)
            preds = classifier.predict([user_inputs])

            if str(preds[0]) in output_message:
                self.message = output_message[str(preds[0])]
            else:
                self.message = "Unexpected error while predicting the output"

        context = {'algo_desc': data['algo_desc'], 'ds_desc': data['ds_desc'],
                   'training_features': data['training_features'], 'graph_image': graph_image,
                   'message': self.message, 'submitbutton': self.submit_button}

        return render(request, self.template_name, context)

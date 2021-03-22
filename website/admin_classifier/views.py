from django.shortcuts import render
import pickle
import pandas as pd
from io import StringIO
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from django.views import View
from pymongo import MongoClient
from bson.binary import Binary
from base64 import b64encode

data = pd.DataFrame({})
client = MongoClient("mongodb+srv://user_1:USER_1@cluster0.0oqke.mongodb.net/<dbname>?retryWrites=true&w=majority")
db = client.get_database('learnml_db')
db_data = db.algorithms

# Create your views here.


class Algorithm(View):

    def get(self, request):
        pass

    def post(self, request):
        pass


class Classification(Algorithm):

    template_name = 'admin_classifier/elements.html'
    context = {}
    feature_list = []
    submit_button = None
    update_button = None
    message = None
    pkl_message = None
    pkl_change_message = None
    csv_message = None
    update_message = None
    accuracy = None
    algo_desc = None
    ds_desc = None

    def get(self, request):

        global db_data
        descriptions = db_data.find_one({'name': 'KNN'})
        self.algo_desc = descriptions['algo_desc']
        self.ds_desc = descriptions['ds_desc']
        self.context = {'algo_desc': self.algo_desc, 'ds_desc': self.ds_desc}
        return render(request, self.template_name, self.context)

    def post(self, request):

        global data, db_data
        descriptions = db_data.find_one({'name': 'KNN'})
        self.algo_desc = descriptions['algo_desc']
        self.ds_desc = descriptions['ds_desc']

        if 'update' in request.POST:
            self.update_button = request.POST.get('update')
            self.algo_desc = str(request.POST.get('algo_desc'))
            self.ds_desc = str(request.POST.get('ds_desc'))
            update_data = {'algo_desc': self.algo_desc, 'ds_desc': self.ds_desc}

            try:
                db_data.update_one({'name': 'KNN'}, {'$set': update_data})
                self.update_message = "Descriptions Updated"
            except:
                self.update_message = "Unexpected error while updating descriptions"
            self.context = {'update_message': self.update_message, 'algo_desc': self.algo_desc,
                            'ds_desc': self.ds_desc}
        elif 'pkl' in request.FILES:
            upload_file = request.FILES['pkl']
            if upload_file.content_type == 'application/octet-stream':
                pkl_obj = pickle.dumps(upload_file)
                mongo_data = {'pkl_data': Binary(pkl_obj), 'upload_method': 'pkl'}
                try:
                    db_data.update_one({"name": "KNN"}, {"$set": mongo_data})
                    self.pkl_message = "File Uploaded"
                except:
                    self.pkl_message = "Unexpected error while uploading pickle file"
            else:
                self.pkl_message = "Invalid File Type"
            self.context = {'pkl_message': self.pkl_message, 'algo_desc': self.algo_desc,
                            'ds_desc': self.ds_desc}
        elif 'pkl_change' in request.POST:
            if request.FILES['graph_image'].content_type == 'image/png' or request.FILES['graph_image'].content_type == 'image/jpeg' or request.FILES['graph_image'].content_type == 'image/jpg':
                pkl_features_temp = str(request.POST.get('pkl_features')).split(',')
                pkl_features = []
                for feature in pkl_features_temp:
                    pkl_features.append(feature.strip())
                pkl_label_temp = str(request.POST.get("pkl_label")).split("\r\n")
                pkl_label = {}
                for label in pkl_label_temp:
                    temp = label.split("=")
                    pkl_label[temp[0].strip()] = temp[1].strip()
                image_file = request.FILES['graph_image']
                image_data = image_file.read()
                encoded_image = str(b64encode(image_data))[2:-1]
                mime = str(image_file.content_type)
                mime = mime + ';' if mime else ';'
                graph_image = "data:%sbase64,%s" % (mime, encoded_image)
                mongo_data = {'training_label': pkl_label, 'training_features': pkl_features,
                              'graph_image': graph_image}
                try:
                    db_data.update_one({"name": "KNN"}, {"$set": mongo_data})
                    self.pkl_change_message = "File Uploaded"
                except:
                    self.pkl_change_message = "Unexpected error while uploading pickle changes"
            else:
                self.pkl_change_message = "Invalid Image Type"
            self.context = {'pkl_change_message': self.pkl_change_message, 'algo_desc': self.algo_desc,
                            'ds_desc': self.ds_desc}
        elif 'csv' in request.FILES:
            upload_file = request.FILES['csv']
            if upload_file.content_type == 'application/vnd.ms-excel':
                string_data = StringIO(upload_file.read().decode('utf-8'))
                data = pd.read_csv(string_data)
                self.feature_list = list(data.columns)
                self.csv_message = "File Uploaded"
            else:
                self.csv_message = "Invalid File Type"
            self.context = {'csv_message': self.csv_message, 'features': self.feature_list, 'algo_desc': self.algo_desc,
                            'ds_desc': self.ds_desc}
        elif 'submit' in request.POST:
            self.submit_button = request.POST.get("submit")
            if request.FILES['csv_image'].content_type == 'image/png' or request.FILES['csv_image'].content_type == 'image/jpeg' or request.FILES['csv_image'].content_type == 'image/jpg':
                n_neighbors = int(request.POST.get("neighbors"))
                leaf_size = int(request.POST.get("leaf"))
                weights = str(request.POST.get("weights"))
                algorithm = str(request.POST.get("algorithm"))
                training_features = list(request.POST.getlist("training_features"))
                training_label = str(request.POST.get("training_label"))
                label_output_temp = str(request.POST.get("label_output")).split("\r\n")
                label_output = {}
                for label in label_output_temp:
                    temp = label.split("=")
                    label_output[temp[0].strip()] = temp[1].strip()
                image_file = request.FILES['csv_image']
                image_data = image_file.read()
                encoded_image = str(b64encode(image_data))[2:-1]
                mime = str(image_file.content_type)
                mime = mime + ';' if mime else ';'
                graph_image = "data:%sbase64,%s" % (mime, encoded_image)

                X = data.loc[:, training_features].values
                y = data.loc[:, training_label].values

                classifier = KNeighborsClassifier(n_neighbors=n_neighbors,
                                                  leaf_size=leaf_size,
                                                  weights=weights,
                                                  algorithm=algorithm)
                classifier.fit(X, y)
                y_pred = classifier.predict(X)

                pkl_obj = pickle.dumps(classifier)
                mongo_data = {'pkl_data': Binary(pkl_obj), 'training_features': training_features,
                              'training_label': label_output, 'graph_image': graph_image, 'upload_method': 'csv'}
                try:
                    db_data.update_one({"name": "KNN"}, {"$set": mongo_data})
                    self.accuracy = round(accuracy_score(y, y_pred)*100, 2)
                    self.message = "Model Successfully Trained"
                except:
                    self.message = "Unexpected error while training the model."
            else:
                self.message = "Invalid Image Type"

            self.context = {'submitbutton': self.submit_button, 'pkl_message': self.pkl_message,
                            'csv_message': self.csv_message, 'accuracy': self.accuracy, 'message': self.message,
                            'algo_desc': self.algo_desc, 'ds_desc': self.ds_desc}

        return render(request, self.template_name, self.context)

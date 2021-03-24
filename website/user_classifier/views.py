from django.shortcuts import render, redirect
import pickle
from django.views import View
import numpy as np
from . import mongodb as mdb
from .forms import CreateUserForm
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.utils.decorators import method_decorator
from django.contrib.auth.models import Group

# Create your views here.
db_data = mdb.access()


class Algorithm(View):

    def get(self, request):
        pass

    def post(self, request):
        pass


@method_decorator(login_required, name='dispatch')
class Classification(Algorithm):

    template_name = 'user_classifier/classification.html'
    message = ""
    submit_button = None

    def get(self, request):

        data = mdb.find(db_data, "KNN")
        context = {'algo_desc': data['algo_desc'], 'ds_desc': data['ds_desc'],
                   'training_features': data['training_features']}

        return render(request, self.template_name, context)

    def post(self, request):

        data = mdb.find(db_data, "KNN")
        graph_image = data['graph_image']
        if data['upload_method'] == 'pkl':
            classifier = pickle.loads(pickle.loads(data['pkl_data']).read())
        else:
            classifier = pickle.loads(data['pkl_data'])
        if 'submit' in request.POST:
            self.submit_button = request.POST.get("submit")
            output_message = data['label_notes']
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


@method_decorator(login_required, name='dispatch')
class Regression(Algorithm):

    template_name = 'user_classifier/regression.html'
    message = ""
    submit_button = None

    def get(self, request):

        data = mdb.find(db_data, "MLR")
        context = {'algo_desc': data['algo_desc'], 'ds_desc': data['ds_desc'],
                   'training_features': data['training_features']}

        return render(request, self.template_name, context)

    def post(self, request):

        data = mdb.find(db_data, "MLR")
        graph_image = data['graph_image']
        if data['upload_method'] == 'pkl':
            regressor = pickle.loads(pickle.loads(data['pkl_data']).read())
        else:
            regressor = pickle.loads(data['pkl_data'])
        if 'submit' in request.POST:
            self.submit_button = request.POST.get("submit")
            user_inputs = np.array(request.POST.getlist('user_inputs')).astype(np.float64)
            preds = regressor.predict([user_inputs])

            self.message = "The predicted profit of the startup is " + str(round(preds[0], 2))

        context = {'algo_desc': data['algo_desc'], 'ds_desc': data['ds_desc'],
                   'training_features': data['training_features'], 'graph_image': graph_image,
                   'message': self.message, 'submitbutton': self.submit_button}

        return render(request, self.template_name, context)


@method_decorator(login_required, name='dispatch')
class Clustering(Algorithm):

    template_name = 'user_classifier/clustering.html'
    message = ""
    submit_button = None

    def get(self, request):

        data = mdb.find(db_data, "KM")
        context = {'algo_desc': data['algo_desc'], 'ds_desc': data['ds_desc'],
                   'training_features': data['training_features']}

        return render(request, self.template_name, context)

    def post(self, request):

        data = mdb.find(db_data, "KM")
        graph_image = data['graph_image']
        if data['upload_method'] == 'pkl':
            classifier = pickle.loads(pickle.loads(data['pkl_data']).read())
        else:
            classifier = pickle.loads(data['pkl_data'])
        if 'submit' in request.POST:
            self.submit_button = request.POST.get("submit")
            output_message = data['label_notes']
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


@method_decorator(login_required, name='dispatch')
class Home(View):

    template_name = 'user_classifier/home.html'
    context = {}

    def get(self, request):
        return render(request, self.template_name, self.context)


class RegisterPage(View):

    template_name = 'user_classifier/register.html'
    context = {}
    form = CreateUserForm()

    def get(self, request):
        if request.user.is_authenticated:
            return redirect('learner_home')
        else:
            self.context = {'form': self.form}
            return render(request, self.template_name, self.context)

    def post(self, request):
        self.form = CreateUserForm(request.POST)

        if self.form.is_valid():
            user = self.form.save()
            username = self.form.cleaned_data.get('username')
            group = Group.objects.get(name='learner')
            user.groups.add(group)
            messages.success(request, "Account was created for " + username)
            return redirect('login')
        else:
            self.context = {'form': self.form}
            return render(request, self.template_name, self.context)


class LoginPage(View):
    template_name = 'user_classifier/login.html'
    context = {}

    def get(self, request):
        if request.user.is_authenticated:
            return redirect('learner_home')
        else:
            return render(request, self.template_name, self.context)

    def post(self, request):
        username = request.POST.get('username')
        password = request.POST.get('password')

        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('learner_home')
        else:
            messages.info(request, "Either Username or Password is incorrect")
            return render(request, self.template_name, self.context)


class LogoutPage(View):

    def get(self, request):
        logout(request)
        return redirect('login')

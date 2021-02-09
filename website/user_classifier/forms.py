from django import forms


# creating a form
class InputForm(forms.Form):
    age = forms.IntegerField(help_text="Enter age")
    salary = forms.IntegerField(help_text="Enter salary")

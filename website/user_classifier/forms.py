from django import forms


# creating a form
class InputForm(forms.Form):
    age = forms.IntegerField(help_text="Enter age of the customer")
    salary = forms.IntegerField(help_text="Enter salary of the customer")

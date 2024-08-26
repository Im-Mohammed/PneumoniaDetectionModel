from django import forms
from .models import getImage

class Test(forms.ModelForm):
    class Meta:
        model = getImage
        fields = '__all__'

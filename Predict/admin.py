from django.contrib import admin
from .models import getImage
# Register your models here.
@admin.register(getImage)
class Imageadmin(admin.ModelAdmin):
    list_display=['name','photo']
from django.db import models

# Create your models here.

class Audio(models.Model):
    id = models.AutoField(max_digits=100)
    name = models.CharField(max_length=100)
    file = models.FileField(upload_to='media/audios')
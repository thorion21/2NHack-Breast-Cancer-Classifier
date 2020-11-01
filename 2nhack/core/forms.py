from django import forms
from django.forms import ModelForm
from django.contrib.auth.forms import UserCreationForm, UserChangeForm
from .models import TCUser, Post, Comment


class PostForm(ModelForm):
    class Meta:
        model = Post
        fields = ['text']


class CommentForm(ModelForm):
    class Meta:
        model = Comment
        fields = ['text']


class TCUserCreationForm(UserCreationForm):

    class Meta(UserCreationForm):
        model = TCUser
        fields = ('email',)


class TCUserChangeForm(UserChangeForm):

    class Meta:
        model = TCUser
        fields = ('email',)

class FileFieldForm(forms.Form):
    file_field = forms.FileField(widget=forms.ClearableFileInput(attrs={'multiple': True}))
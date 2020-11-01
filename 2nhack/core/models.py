from django.db import models
from django.core.validators import RegexValidator
from django.contrib.auth.models import AbstractBaseUser
from django.contrib.auth.models import PermissionsMixin
from django.utils.translation import gettext_lazy as _
from django.utils import timezone

from .managers import TCUserManager


class TCUser(AbstractBaseUser, PermissionsMixin):
    email = models.EmailField(_('email address'), unique=True)
    first_name = models.CharField(max_length=255, null=True, blank=False)
    last_name = models.CharField(max_length=255, null=True, blank=False)

    phone_regex = RegexValidator(
        regex=r'^(\+4|)?(07[0-8]{1}[0-9]{1}|02[0-9]{2}|03[0-9]{2}){1}?(\s|\.|\-)?([0-9]{3}(\s|\.|\-|)){2}$',
        message="Numarul de telefon trebuie sa fie de forma 07XXXXXXXX sau +407XXXXXXXX")
    phone_number = models.CharField(validators=[phone_regex], max_length=14, blank=True)  # validators should be a list

    is_staff = models.BooleanField(default=False)
    is_active = models.BooleanField(default=True)
    date_joined = models.DateTimeField(default=timezone.now)

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = []

    objects = TCUserManager()

    class Meta:
        verbose_name = 'TCUser'
        verbose_name_plural = 'TCUsers'

    def __str__(self):
        return self.email


class Post(models.Model):
    text = models.CharField(max_length=200)
    image_file_paths = models.TextField(blank=True, null=True)
    created_by = models.ForeignKey(
        TCUser,
        on_delete=models.CASCADE,
        related_name='posts'
    )
    created_at = models.DateField(auto_now_add=True)

    def paths_as_list(self):
        return set(self.image_file_paths.split('|')[:-1])

    def verdict(self):
        return str(self.image_file_paths.split('|')[-1])

    def __str__(self):
        return 'Post ' + self.text + ' created on ' + self.created_at.strftime('%m/%d/%Y, %H:%M:%S')


class Comment(models.Model):
    text = models.CharField(max_length=75)
    post = models.ForeignKey(
        Post,
        on_delete=models.CASCADE,
        related_name='comments'
    )
    created_by = models.ForeignKey(
        TCUser,
        on_delete=models.CASCADE,
        related_name='comments'
    )
    created_at = models.DateField(auto_now_add=True)
    updated_at = models.DateField(auto_now=True)

    def __str__(self):
        return 'Comment ' + self.text + ' created on ' + self.created_at.strftime('%m/%d/%Y, %H:%M:%S')
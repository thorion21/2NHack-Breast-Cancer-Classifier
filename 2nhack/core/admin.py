from django.contrib import admin
from django.contrib.auth.admin import UserAdmin

from .forms import TCUserCreationForm, TCUserChangeForm
from .models import TCUser, Post, Comment


class PostAdmin(admin.ModelAdmin):
    search_fields = ['text', 'created_by']
    list_filter = ('created_by',)
    list_display = ['created_by', 'text', 'created_at']


class CommentAdmin(admin.ModelAdmin):
    search_fields = ['text', 'post']
    list_filter = ('created_by', 'post')
    list_display = ['created_by', 'text', 'post']


class TCUserAdmin(UserAdmin):
    add_form = TCUserCreationForm
    form = TCUserChangeForm
    model = TCUser
    list_display = ('email', 'last_name', 'first_name', 'phone_number',)
    list_filter = ('last_name', 'first_name',)
    fieldsets = (
        (None, {'fields': ('email', 'password', 'last_name', 'first_name', 'phone_number',)}),
        ('Permissions', {'fields': ('is_staff', 'is_active')}),
    )
    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('email', 'password1', 'password2', 'is_staff', 'is_active')}
        ),
    )
    search_fields = ('email', 'last_name', 'first_name', 'phone_number',)
    ordering = ('email', 'last_name', 'first_name', 'phone_number',)

admin.site.register(TCUser, TCUserAdmin)
admin.site.register(Post, PostAdmin)
admin.site.register(Comment, CommentAdmin)

from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.shortcuts import render
from django.views.generic.edit import (
    FormView
)
from .forms import FileFieldForm, CommentForm
from django.views.generic import (
    TemplateView, View, RedirectView, ListView, DetailView
)
from django.shortcuts import redirect
import os
from PIL import Image

from keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
from keras.preprocessing.image import img_to_array

from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import (
    REDIRECT_FIELD_NAME,
    login as auth_login,
    logout as auth_logout,
)
from django.views.decorators.cache import never_cache
from django.views.decorators.csrf import csrf_protect
from django.views.decorators.debug import sensitive_post_parameters
from django.utils.decorators import method_decorator
from django.utils.http import is_safe_url
from django.contrib.auth.mixins import LoginRequiredMixin
from .models import Post, Comment
from django.core.exceptions import ObjectDoesNotExist
from django.contrib import messages


FILES_DIR = 'userupload'
RESULTS_DIR = 'core/static/images/results'
ML_TRUE = [0, 1]
ML_FALSE = [1, 0]


def run_ml_algo():
    # load model
    loaded_model = load_model('cancernet.h5')
    # summarize model.
    # loaded_model.summary()
    # load dataset

    X = load_img("mlpass_test.png", target_size=(48, 48))
    X = img_to_array(X)
    X = np.expand_dims(X, axis=0)
    print(X.shape)

    Y = loaded_model.predict(X, batch_size=1)
    print("Algorithm results", Y, type(Y), Y[0], type(Y[0]), "list", list(Y[0]))
    print("Rezz", list(Y[0]) == ML_TRUE, list(Y[0]) == ML_FALSE)

    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    score = loaded_model.evaluate(np.array(X), np.array(Y), verbose=0)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))

    return list(Y[0]) == ML_TRUE


class NotificationsView(LoginRequiredMixin, ListView):
    template_name = 'notifications.html'
    model = Post


class NotificationsDetailPage(DetailView):
    model = Post
    template_name = "notifications_detail_view.html"

    def get_context_data(self, **kwargs):
        # Call the base implementation first to get a context
        context = super(NotificationsDetailPage, self).get_context_data(**kwargs)
        context['form'] = CommentForm()
        # for img in [x for x in current_post.image_file_paths.split('|')]:
        #     messages.add_message(request, messages.INFO, img)
        return context

    def post(self, request, pk):
        try:
            current_post = Post.objects.get(pk=pk)

        except ObjectDoesNotExist:
            error_string = "Post not found"
            return render(request, 'error_page.html', {'text': error_string})

        form = CommentForm(request.POST)
        if form.is_valid():
            text = form.cleaned_data['text']
            new_comment = Comment(text=text, created_by=request.user, post=current_post)
            new_comment.save()

        return redirect('notifications_detail_page_view', pk=pk)


# Imaginary function to handle an uploaded file.
def handle_uploaded_file(file):
    print("Am apelat functia!", file)
    dir_path = FILES_DIR + '/'
    path = dir_path + str(file)
    res_dir_path = RESULTS_DIR + '/'
    res_path = res_dir_path + 'res_' + str(file)
    print('FINAL PATH = ', path)
    print('RESULTS FINAL PATH = ', res_path)

    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

    if not os.path.isdir(res_dir_path):
        os.mkdir(res_dir_path)

    with open(path, 'wb+') as destination:
        for chunk in file.chunks():
            destination.write(chunk)

    return res_path, run_ml_algo()


class InvestigatieView(LoginRequiredMixin, FormView):
    form_class = FileFieldForm
    template_name = 'investigatie.html'  # Replace with your template.
    success_url = '../results'

    def post(self, request, *args, **kwargs):
        form_class = self.get_form_class()
        form = self.get_form(form_class)
        files = request.FILES.getlist('file_field')
        verdict = False
        if form.is_valid():
            filename = str(files[0])
            is_archive = filename.split('.')[1] == 'zip'

            ml_train_set_filepaths = []

            if is_archive:
                # Handle unzip and processing and return list with file names
                # unzip
                # get list of filepaths
                # run ml_algo on batch
                pass
            else:
                for f in files:
                    path, iz_cansur = handle_uploaded_file(f)
                    ml_train_set_filepaths.append(path)

                    if iz_cansur:
                        verdict = True

            request.session['ml_train_set_filepaths'] = ml_train_set_filepaths
            request.session['ml_verdict'] = verdict

            return self.form_valid(form)
        else:
            return self.form_invalid(form)


class AlegeMediciView(TemplateView):
    template_name = "alege_medici.html"


class CalendarView(TemplateView):
    template_name = "calendar.html"


class ResultsView(View):

    def get(self, request, *args, **kwargs):
        print("RESULTATUL A FOST GENERAT")
        ml_train_set_filepaths = request.session.get('ml_train_set_filepaths')
        raw_string = '|'.join([x[12:] for x in ml_train_set_filepaths])
        ml_verdict = request.session.get('ml_verdict')
        raw_string += "|IDC+" if ml_verdict else "|IDC-"
        print(ml_train_set_filepaths)
        ml_train_set_filepaths = ['images/results/' + x.split('/')[-1] for x in ml_train_set_filepaths]
        from datetime import datetime
        name = "Consultație amănunțită pacient " + datetime.now().strftime("RO_%Y%m%d_%H%M%S")
        objs = Post.objects.filter(text=name)
        if not objs:
            p = Post(text=name, image_file_paths=raw_string, created_by=self.request.user)
            p.save()


        return render(request, 'results.html', {'data': ml_train_set_filepaths, 'verdict': ml_verdict, 'raw_string': raw_string})


class HomepageView(TemplateView):
    template_name = 'index.html'


class LoginView(FormView):
    success_url = ''
    form_class = AuthenticationForm
    redirect_field_name = REDIRECT_FIELD_NAME
    template_name = 'login.html'

    def get(self, request, *args, **kwargs):
        if request.user.is_authenticated:
            return redirect('homepage_view')

        return super(LoginView, self).get(request, args, kwargs)

    @method_decorator(sensitive_post_parameters("password"))
    @method_decorator(csrf_protect)
    @method_decorator(never_cache)
    def dispatch(self, request, *args, **kwargs):
        request.session.set_test_cookie()

        return super(LoginView, self).dispatch(request, *args, **kwargs)

    def form_valid(self, form):
        auth_login(self.request, form.get_user())

        if self.request.session.test_cookie_worked():
            self.request.session.delete_test_cookie()

        return super(LoginView, self).form_valid(form)

    def get_success_url(self):
        redirect_to = self.request.GET.get(self.redirect_field_name)
        if not is_safe_url(
            url=redirect_to, allowed_hosts=self.request.get_host()
        ):
            redirect_to = self.success_url
        return redirect_to


class LogoutView(LoginRequiredMixin, RedirectView):
    url = "/"

    def get(self, request, *args, **kwargs):
        auth_logout(request)
        return super(LogoutView, self).get(request, *args, **kwargs)

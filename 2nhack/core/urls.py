from django.urls import path, re_path

from .views import (
    HomepageView, InvestigatieView, ResultsView, LoginView, LogoutView, NotificationsView, NotificationsDetailPage,
    AlegeMediciView, CalendarView
)

urlpatterns = [
    path('', HomepageView.as_view(), name='homepage_view'),
    path('investigatie/', InvestigatieView.as_view(), name='investigatie_view'),
    path('results/', ResultsView.as_view(), name='results_view'),
    path('login/', LoginView.as_view(), name='login_view'),
    path('logout/', LogoutView.as_view(), name='logout_view'),
    path('notifications/', NotificationsView.as_view(), name='notifications_view'),
    path('notifications/<int:pk>', NotificationsDetailPage.as_view(), name='notifications_detail_page_view'),
    path('alege_medici/', AlegeMediciView.as_view(), name='alege_view'),
    path('calendar/', CalendarView.as_view(), name='calendar_view'),
]
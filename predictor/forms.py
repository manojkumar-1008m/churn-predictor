from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

YES_NO = [("Yes", "Yes"), ("No", "No")]


class RegisterForm(UserCreationForm):
    email = forms.EmailField(required=True)
    phone_number = forms.RegexField(
        regex=r"^\+?[0-9\-\s\(\)]{7,20}$",
        required=True,
        error_messages={"invalid": "Enter a valid phone number."},
        label="Phone no",
        help_text="Use digits and optional +, -, spaces, or parentheses.",
    )

    class Meta(UserCreationForm.Meta):
        model = User
        fields = ("username", "email", "phone_number", "password1", "password2")

    def clean_email(self):
        email = self.cleaned_data["email"].strip().lower()
        if User.objects.filter(email__iexact=email).exists():
            raise forms.ValidationError("An account with this email already exists.")
        return email

    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data["email"]
        if commit:
            user.save()
        return user


class ChurnPredictionForm(forms.Form):
    gender = forms.ChoiceField(choices=[("Female", "Female"), ("Male", "Male")])
    SeniorCitizen = forms.TypedChoiceField(choices=[(0, "No"), (1, "Yes")], coerce=int)
    Partner = forms.ChoiceField(choices=YES_NO)
    Dependents = forms.ChoiceField(choices=YES_NO)
    tenure = forms.IntegerField(min_value=0, max_value=120)
    PhoneService = forms.ChoiceField(choices=YES_NO)
    MultipleLines = forms.ChoiceField(
        choices=[("No", "No"), ("Yes", "Yes"), ("No phone service", "No phone service")]
    )
    InternetService = forms.ChoiceField(
        choices=[("DSL", "DSL"), ("Fiber optic", "Fiber optic"), ("No", "No")]
    )
    OnlineSecurity = forms.ChoiceField(
        choices=[("No", "No"), ("Yes", "Yes"), ("No internet service", "No internet service")]
    )
    OnlineBackup = forms.ChoiceField(
        choices=[("No", "No"), ("Yes", "Yes"), ("No internet service", "No internet service")]
    )
    DeviceProtection = forms.ChoiceField(
        choices=[("No", "No"), ("Yes", "Yes"), ("No internet service", "No internet service")]
    )
    TechSupport = forms.ChoiceField(
        choices=[("No", "No"), ("Yes", "Yes"), ("No internet service", "No internet service")]
    )
    StreamingTV = forms.ChoiceField(
        choices=[("No", "No"), ("Yes", "Yes"), ("No internet service", "No internet service")]
    )
    StreamingMovies = forms.ChoiceField(
        choices=[("No", "No"), ("Yes", "Yes"), ("No internet service", "No internet service")]
    )
    Contract = forms.ChoiceField(
        choices=[("Month-to-month", "Month-to-month"), ("One year", "One year"), ("Two year", "Two year")]
    )
    PaperlessBilling = forms.ChoiceField(choices=YES_NO)
    PaymentMethod = forms.ChoiceField(
        choices=[
            ("Electronic check", "Electronic check"),
            ("Mailed check", "Mailed check"),
            ("Bank transfer (automatic)", "Bank transfer (automatic)"),
            ("Credit card (automatic)", "Credit card (automatic)"),
        ]
    )
    MonthlyCharges = forms.FloatField(min_value=0)
    TotalCharges = forms.FloatField(min_value=0)


class ModelUploadForm(forms.Form):
    model_file = forms.FileField(help_text="Upload a .joblib pipeline file")

    def clean_model_file(self):
        uploaded = self.cleaned_data["model_file"]
        if not uploaded.name.lower().endswith(".joblib"):
            raise forms.ValidationError("Only .joblib model files are supported.")
        return uploaded


class DatasetUploadForm(forms.Form):
    dataset_file = forms.FileField(help_text="Upload a dataset file (.csv, .xlsx, .xls)")

    def clean_dataset_file(self):
        uploaded = self.cleaned_data["dataset_file"]
        allowed_extensions = (".csv", ".xlsx", ".xls")
        if not uploaded.name.lower().endswith(allowed_extensions):
            raise forms.ValidationError("Only .csv, .xlsx, and .xls files are supported.")
        return uploaded

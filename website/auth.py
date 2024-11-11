from flask import Blueprint, render_template, request, flash
from .summarizer import summarize

auth = Blueprint('auth', __name__)

@auth.route('/login', methods=['GET', 'POST'])
def login():
    return render_template("login.html", boolean= True)

@auth.route('/logout')
def logout():
    return render_template("logout.html")

@auth.route('/summarize')
def summarize_home():
    return render_template("summarize.html")

@auth.route('/about')
def about():
    return render_template("about.html")


@auth.route('/summarize_post', methods=['POST'])
def summarizer_route():
    inputText=request.form.get("inputText")
    print(inputText)
    summarized_sentences=summarize(inputText)
    print(summarized_sentences)
    return render_template("summarize.html", summarized_text=summarized_sentences)


@auth.route('/sign-up', methods=['GET', 'POST'])
def sign_up():
    if request.method == 'POST':
        email = request.form.get('email')
        first_name = request.form.get('firstName')
        password1 = request.form.get('password1')
        password2 = request.form.get('password2')
        if len(email) < 4:
            flash('Email must be greater than three characters.', category = 'error')
        elif len(first_name) < 2:
            flash('First name must be greater than one character.', category = 'error')
        elif password1 != password2:
            flash('Passwords must match.', category = 'error')
        elif len(password1) < 7:
            flash('Password must be at least seven characters.', category = 'error')
        else:
            flash('Account created!', category = 'success')
    return render_template("sign_up.html")

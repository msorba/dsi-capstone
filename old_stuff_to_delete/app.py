import os
from flask import Flask, render_template, request, session
import functions as fn
from PIL import Image
from flask_session import Session

app = Flask(__name__)   
sess = Session()
# Check Configuration section for more details
SESSION_TYPE = 'filesystem'
app.secret_key = 'supersecret1234'
app.config['SESSION_TYPE'] = SESSION_TYPE


app.config.from_object(__name__)
sess.init_app(app)
app.run()

# @app.route('/upload')
# def set():
#     session['key'] = 'sess12345'
#     return 'ok'

# @app.route('/results')
# def get():
#     return session.get('key', 'not set')



# app = Flask(__name__)

UPLOAD_FOLDER = os.path.join('static','uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

RESULTS_FOLDER = os.path.join('static','results')
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER


@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response

@app.route('/')
def hello_world():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['image']
    f = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    
    # add your custom code to check that the uploaded file is a valid image and not a malicious file (out-of-scope for this post)
    file.save(f)
    session['uploaded_image'] = f
    return render_template('index.html',display_image = f)

@app.route('/results', methods=['GET'])
def get_results():
    uploaded_image = session.get('uploaded_image',None)
    # img_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_image)
    img = Image.open(uploaded_image)
    img = fn.threshold_image(img)
    _,wrinkle_class = fn.get_wrinkle_class(img, n_clusters=2)
    if fn.count_spokes(wrinkle_class) > 0:
        genotype = 'phz'
    else:
        genotype = 'WT'
    fn.plot_wrinkle_class(wrinkle_class)

    f = os.path.join(app.config['RESULTS_FOLDER'], 'wrinkle.png')
    return render_template('results.html',display_image = f , genotype_display = genotype)
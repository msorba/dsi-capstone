import os
from flask import Flask, render_template, request, session, jsonify
from PIL import Image
from flask_session import Session
import json
from app_functions import *
from image_processing_functions import *

app = Flask(__name__)
sess = Session()
# Check Configuration section for more details
SESSION_TYPE = 'filesystem'
app.secret_key = '<insert_some_secret_key_here>'
app.config['SESSION_TYPE'] = SESSION_TYPE


app.config.from_object(__name__)
sess.init_app(app)
# app.run()


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
	f = os.path.join(app.config['UPLOAD_FOLDER'], "sample.jpg")
	session['uploaded_image'] = os.path.join('mysite',f)
	return render_template("index.html", display_image=f)

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['image']
    f_save = os.path.join('mysite',app.config['UPLOAD_FOLDER'], file.filename)
    f_show = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(f_save)
    session['uploaded_image'] = f_save
    return render_template('index.html',display_image = f_show)

@app.route('/results', methods=['GET', 'POST'])
def get_results():
	uploaded_image = session.get('uploaded_image',None)
	# alpha = request.form["sliderGlare"]
	inputs = jsonify(request.form).response[0]
	inputs_dict = json.loads(inputs)


	bg = int(inputs_dict.get("background",40))
	al = int(inputs_dict.get('alpha',80))
	bt = int(inputs_dict.get('beta',110))
	redCheck = int(inputs_dict.get('redCheck',0))
	im_path = os.path.join('mysite',uploaded_image)
	image = Image.open(uploaded_image)
	wrinkle_labels,processed_image = get_wrinkles(image, resize = 500, mask=bg, background_is_black = True, is_red=redCheck, alpha=al, beta = bt)
	plot_and_save(processed_image,uploaded_image, suffix = "wrinkle")


	perc_wrinkle = perc_wrinkled(wrinkle_labels)

	spoke_img, spoke_count, spoke_median_length, spoke_median_dist_center = detect_spokes(image, img_wrinkle=processed_image)
	plot_and_save(spoke_img,uploaded_image, suffix = "spokes")

	pw_message = "Percent Wrinkled: %s " % perc_wrinkle
	spoke_count_message = "Spoke Count: %d" % spoke_count
	spoke_ratio_message = "Spoke Ratio: %f2" %(spoke_median_length/spoke_median_dist_center)


	im = uploaded_image.split('/')[-1].split('.')[0]
	f1 = os.path.join(app.config['RESULTS_FOLDER'], im+'_wrinkle.png')
	f2 = os.path.join(app.config['RESULTS_FOLDER'], im+'_spokes.png')
	f = os.path.join(app.config['RESULTS_FOLDER'], im)
	return render_template('results.html',display_wrinkle_image = f1, display_spoke_image = f2,\
		message1 = pw_message, message2 = spoke_count_message, message3 = spoke_ratio_message, \
		bg = bg, al = al, bt = bt, redCheck = redCheck)


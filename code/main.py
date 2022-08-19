import pickle
from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
import os
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
from PIL.Image import Image as PilImage
import random
import json
import matplotlib.patches as patches
import math
from skimage import io
import skimage.feature
import cv2
from custom_class import custom_class


app = Flask(__name__)
upload_folder = "./static/img"
allowed_extensions = {"jpg"}


app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = upload_folder


@app.route("/", methods=["GET"])
def index():
    """
    Render the main landing page on arrival
    """
    return render_template("index.html")


@app.route("/index.html", methods=["GET"])
def main_index():
    """
    Render the main landing page from other pages
    """
    return render_template("index.html")


@app.route("/single-post.html", methods=["GET"])
def single_post():
    """
    Simple page explaining model choice, mostly empty
    """
    return render_template("single-post.html")


@app.route("/data_exploration.html", methods=["GET"])
def data_exploration():
    """
    Simple page going through data exploration, mostly empty
    """
    return render_template("data_exploration.html")


def allowed_file(filename):
    """
    Function verifying the file extension is within allowed list
    """
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions


@app.route("/new_test.html", methods=["GET", "POST"])
def upload_file():
    """
    Render page which allows the user to upload a test file.
    """
    if request.method == "POST":
        # check if the post request has the file part
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["file"]
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], "test_image.jpg"))
    return render_template("new_test.html")


@app.route("/prediction.html", methods=["GET"])
def prediction():
    """
    Function verifying the file extension is within allowed list.
    """
    model = custom_class()
    result = model.predict()
    return render_template("prediction.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

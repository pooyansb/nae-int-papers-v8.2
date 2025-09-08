from flask import Blueprint, render_template, redirect, url_for

home_bp = Blueprint("home", __name__)

@home_bp.route("/", methods=["GET"])
def home():
    #return render_template("home.html")
    return redirect(url_for("sankey.sankey"))

from flask import Blueprint, render_template

sankey_bp = Blueprint("sankey", __name__)

@sankey_bp.route("/sankey", methods=["GET"])
def sankey():
    return render_template("sankey.html")

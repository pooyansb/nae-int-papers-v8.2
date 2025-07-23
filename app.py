from flask import Flask
from routes.home import home_bp
from routes.sankey import sankey_bp
from routes.search import search_bp
from routes.about import about_bp
from config import Config

def create_app():
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config.from_object(Config)

    app.register_blueprint(home_bp)
    app.register_blueprint(sankey_bp)
    app.register_blueprint(search_bp)
    app.register_blueprint(about_bp)

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)

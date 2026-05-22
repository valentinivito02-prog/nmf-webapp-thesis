# TEST CLAUDE ACTIVE
from flaskr import create_app

flask_app = create_app()

if __name__ == '__main__':
    flask_app.run(debug=True, port=5000)

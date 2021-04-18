from flask import Flask, request, redirect, flash, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename
from flask_restful import Api
import os
from detection1 import run_yolo_prediction
import cloudinary
import cloudinary.uploader

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
api = Api(app)

app.secret_key = b'secret'

BASE_PATH = os.path.abspath("./")

ALLOWED_EXTENSION = {'mp4', 'jpg', 'png', 'jpeg'}

cloudinary.config(
  cloud_name= "YOUR_CLOUD_NAME",
  api_key= "YOUR_API_KEY",
  api_secret= "YOUR_API_SECRET"
)

def allowed_files(filename: str) -> bool:
    print('.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSION)
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSION

@app.route('/video', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file')
            return redirect(request.url)
    
    file = request.files.get('file')

    if file.filename == '':
        flash('No file was select')
        return redirect(request.url)

    if file and allowed_files(file.filename):
      filename = secure_filename(file.filename)
      print(filename)
      file.save(os.path.join(BASE_PATH, 'detection/input/', filename))
      flash('File Uploaded Successfully')
      run_yolo_prediction({
        "video": filename,
        "save": True
      })

      upload_result = cloudinary.uploader.upload("./detection/output/out_{}".format(filename), public_id = "cv_result", resource_type="video")
      return Response(upload_result['url'][:-4] + ".webm")
    return abort(404)

if __name__ == '__main__':
  app.run(debug=True)







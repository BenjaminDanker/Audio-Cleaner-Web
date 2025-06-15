from flask import Flask, render_template, request, send_file, after_this_request
import os
import tempfile
import shutil

from processing import denoise_video, DEFAULT_ATTEN_DB

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded = request.files.get('video')
        if not uploaded:
            return 'No video uploaded', 400
        try:
            atten = int(request.form.get('atten', DEFAULT_ATTEN_DB))
        except ValueError:
            atten = DEFAULT_ATTEN_DB
        tmpdir = tempfile.mkdtemp()
        input_path = os.path.join(tmpdir, uploaded.filename)
        uploaded.save(input_path)
        output_path = os.path.join(tmpdir, 'output.mp4')
        denoise_video(input_path, output_path, atten_lim_db=atten)

        response = send_file(output_path, as_attachment=True, download_name='denoised.mp4')
        response.call_on_close(lambda: shutil.rmtree(tmpdir, ignore_errors=True))
        return response
    return render_template('index.html', default_atten=DEFAULT_ATTEN_DB)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)

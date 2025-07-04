from flask import Flask, render_template, request, send_file, after_this_request
import os
from video_handler import DEFAULT_ATTEN_DB
from video_handler import VideoProcessor

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        uploaded_file = request.files.get("video")
        if not uploaded_file or not uploaded_file.filename:
            return "No video uploaded or filename missing", 400

        try:
            atten = int(request.form.get("atten", DEFAULT_ATTEN_DB))
        except ValueError:
            atten = DEFAULT_ATTEN_DB

        processor = VideoProcessor(uploaded_file, atten)

        try:
            output_path = processor.process()
            response = send_file(output_path, as_attachment=True, download_name='denoised.mp4')

            @after_this_request
            def cleanup_after_request(response_param):
                processor.schedule_cleanup(app.logger)
                return response_param

            return response
        except Exception as e:
            app.logger.error(f"Error during video processing or file sending: {e}")
            processor.immediate_cleanup(app.logger)
            return "Error processing video", 500

    return render_template('index.html', default_atten=DEFAULT_ATTEN_DB)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
